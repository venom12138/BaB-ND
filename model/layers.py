import torch
import torch.nn as nn

class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        # self.norm = nn.LayerNorm(normalized_shape=[output_size])

    def forward(self, x, res=None):
        '''
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        '''
        if self.residual:
            x = self.relu(self.linear(x) + res)
        else:
            assert res == None
            x = self.relu(self.linear(x))

        return x # self.norm(x)

class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        # print(x.size())
        return self.model(x)

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
        return self.model(x)

class GNN_layer(nn.Module):
    def __init__(self, config):
        super(GNN_layer, self).__init__()
        self.config = config['train']["model_param"]
        nf_particle = self.config['nf_particle']
        nf_relation = self.config['nf_relation']
        self.nf_particle = nf_particle
        nf_effect = nf_particle
        self.nf_relation = nf_relation

        if config['task_name'] == 'pushing_rope':
            # 3d or 2d
            dim_of_work_space = config['data']['state_dim'] // config['data']['max_nobj']
        else:
            dim_of_work_space = 2
        attr_dim = config['train']['n_history'] * dim_of_work_space * 2 # action+state
        state_dim = config['train']['n_history'] * dim_of_work_space # state
        self.residual = True

        # (1) particle attr (2) state
        self.particle_encoder = ParticleEncoder(attr_dim, nf_particle, nf_effect)

        # (1) sender attr (2) receiver attr (3) state receiver (4) state_diff (5) relation attr
        self.relation_encoder = RelationEncoder(2*attr_dim + state_dim,
            nf_relation, nf_relation)
        
        # (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(nf_relation + 2 * nf_effect, nf_relation)

        # (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(2 * nf_effect, nf_effect, residual=True)


    def forward(self, particle_effect, attr, state, Rr, Rs, pstep=2):
        # attr: (bs, N, attr_dim)
        # state: (bs, N, state_dim)
        # Rr: (bs, n_relations, N)
        # Rs: (bs, n_relations, N)
        Rrp = Rr.transpose(1, 2) # (bs, N, n_relations)
        Rsp = Rs.transpose(1, 2) # (bs, N, n_relations)

        # receiver_attr, sender_attr
        attr_r_rel = Rr.bmm(attr) # (bs, n_relations, attr_dim)
        attr_s_rel = Rs.bmm(attr) # (bs, n_relations, attr_dim)

        # receiver_state, sender_state
        state_r_rel = Rr.bmm(state) # B x n_relations x state_dim
        state_s_rel = Rs.bmm(state) # B x n_relations x state_dim

        # particle encode
        particle_encode = self.particle_encoder(attr) # (bs, N, nf_effect)

        # calculate relation encoding
        relation_encode = self.relation_encoder(
            torch.cat([attr_r_rel, attr_s_rel, state_r_rel - state_s_rel], 2)) # (bs, n_relations, nf_effect)
        
        # pstep aggregate
        for i in range(pstep):
            # aggregate information from adjacent particles
            # particle -> edge
            receiver_effect = Rr.bmm(particle_effect) # (bs, n_relations, nf_effect)
            sender_effect = Rs.bmm(particle_effect) # (bs, n_relations, nf_effect)

            # calculate relation effect
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, receiver_effect, sender_effect], 2)) # (bs, n_relations, nf_effect)

            # calculate particle effect by aggregating relation effect
            # edge -> particle
            # DGL function
            effect_rel_agg = Rrp.bmm(effect_rel) # B x N x nf_effect

            # calculate particle effect
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                res=particle_effect) # (bs, N, nf_effect)

        # particle_pred = self.particle_predictor(particle_effect)

        return particle_effect

class MLP_layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_layer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)
