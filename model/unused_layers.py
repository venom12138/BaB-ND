class GNN_layer(nn.Module):
    def __init__(self, config):
        super(GNN_layer, self).__init__()
        self.config = config["GNN"]
        nf_particle = self.config['nf_particle']
        nf_relation = self.config['nf_relation']

        self.nf_particle = self.config['nf_particle']
        self.residual = True

        # (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(nf_relation + 2 * nf_particle, nf_relation)

        # (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(nf_particle+nf_relation, nf_particle, residual=True)


    def forward(self, node_feat, edge_index, edge_feat, particle_effect, verbose=0):
        # node_feat: (N, nf_particle)
        # state_norm: (N, nf_particle)
        # edge_feat:(E, nf_relation)
        for _ in range(2):
            effect_i = particle_effect[edge_index[0]]
            effect_j = particle_effect[edge_index[1]] # E, dim
            # print(f"effect_i: {effect_i.shape}, effect_j: {effect_j.shape} edge_feat: {edge_feat.shape}")
            effect_rel = self.relation_propagator(
                torch.cat([edge_feat, effect_i, effect_j], dim=-1))
            aggr_out = torch_scatter.scatter(src=effect_rel, index=edge_index[0], dim=0) # N, nf_relation
            
            particle_effect = self.particle_propagator(
                    torch.cat([node_feat, aggr_out], dim=-1),
                    res=particle_effect) # (N, nf_particle)
        
        return particle_effect

class PyG_GNNBlock(nn.Module):
    def __init__(self, config,):
        super(PyG_GNNBlock, self).__init__()
        self.config = config        
        self.blocks = nn.ModuleList()
        self.particle_encoder = nn.Sequential(*[
            nn.Linear(config['n_history'] * 2, config['GNN']['nf_particle']),
            nn.ReLU()
        ])
        self.relation_encoder = nn.Sequential(*[
            nn.Linear(config['n_history'] * 2, config['GNN']['nf_relation']),
            nn.ReLU()
        ])
        for _ in range(config['GNN']['layers']):
            self.blocks.append(GNN_layer(config))
        self.predictor = nn.Linear(config['GNN']['nf_particle'], 2)
    
    def forward(self, attr, state_norm, edge_index):
        # attr: (bs, N, attr_dim)
        # state: (bs, N, state_dim)
        # calculate particle encoding
        
        node_feat = self.particle_encoder(attr)
        particle_effect = torch.zeros_like(node_feat)
        # into batch
        data_list = []
        for bs_idx in range(attr.shape[0]):
            batch_edge_feat = state_norm[bs_idx, edge_index[bs_idx][0]] - state_norm[bs_idx, edge_index[bs_idx][1]]
            batch_edge_feat = self.relation_encoder(batch_edge_feat)
            data_list.append(Data(edge_attr=batch_edge_feat, particle_effect=particle_effect[bs_idx], \
                node_feat=node_feat[bs_idx], edge_index=edge_index[bs_idx], num_nodes=attr.shape[1]))
            
        g_batch = Batch.from_data_list(data_list)
        # pstep aggregate
        for i in range(len(self.blocks)):
            particle_effect = self.blocks[i](node_feat=g_batch.node_feat, edge_index=g_batch.edge_index, \
                edge_feat=g_batch.edge_attr, particle_effect=g_batch.particle_effect)
            g_batch.particle_effect = particle_effect
        
        out_data_list = g_batch.to_data_list()
        particle_effect_out = torch.stack([out_data_list[i].particle_effect for i in range(len(out_data_list))])
        
        return self.predictor(particle_effect_out) # B, N, 2

def preprocess(self, model_input):
    """
    input_dict: dictionary with keys 'observation' and 'action'
    """
    # reshape    
    # B, n_his, N, 2
    state_ori = model_input['state'].reshape(-1, self.n_history, self.state_dim//2, 2) # [B, n_his, N, 2]
    action = model_input['action'] # [B, n_his, action_dim]
    state = state_ori.clone().detach()
    B, n_his, N, _ = state.shape
    assert action.shape[-1] == state.shape[-1]
    
    # generate edge_index
    thresh = (self.config["data"]["obj_size"]*2) / self.config["data"]["scale"] + self.config["train"]["GNN"]["neighbor_radius"]
    s_receiv = state[:, -1].reshape(B, 1, N, 2).repeat(1, N, 1, 1)
    s_sender = state[:, -1].reshape(B, N, 1, 2).repeat(1, 1, N, 1)
    dis = torch.sum((s_sender - s_receiv)**2, -1)
    adj_matrix = ((dis - thresh) < 0).float()
    edge_index = []
    for b in range(B):
        positive_indices = torch.argwhere(adj_matrix[b] > 0)
        edge_indices = positive_indices.T  # [2, #edges]
        edge_index.append(edge_indices.to(state.device))
    
    # generate attr for each particle
    # only consider the particles that are within the pusher's reach
    pos_diff_to_end_pusher = state - action.reshape(B, n_his, 1, action.shape[-1]) # [B, n_his, N, 2]
    pusher_dir = action / torch.norm(action, dim=-1, keepdim=True)
    compensate_len = (self.config["data"]["obj_size"] + self.config["data"]["pusher_size"][1] / 2) / self.config["data"]["scale"]
    compensate_wid = (self.config["data"]["obj_size"] + self.config["data"]["pusher_size"][0] / 2) / self.config["data"]["scale"]
    push_len = torch.norm(action, dim=-1, keepdim=True) + compensate_len # B, n_his, 2
    push_len = push_len.unsqueeze(-2)
    push_wid = compensate_wid
    pos_diff_proj = state * pusher_dir.reshape(B, n_his, 1, action.shape[-1])
    pos_diff_orth = (state - pos_diff_proj)
    pos_diff_proj_len = pos_diff_proj.sum(dim=-1, keepdim=True)  # B, n_his, N, 1
    pos_diff_orth_len = torch.abs(pos_diff_orth.sum(dim=-1, keepdim=True)) # B, n_his, N, 1
    pos_diff_l_mask = ((pos_diff_proj_len < push_len) & (pos_diff_proj_len > -compensate_len)).to(torch.float32)
    pos_diff_w_mask = (pos_diff_orth_len < push_wid).to(torch.float32)
    attr = pos_diff_to_end_pusher * pos_diff_l_mask * pos_diff_w_mask # [B, n_his, N, 2]
    # print(f"action:{action.shape}")
    # print(f"push_len:{push_len[:2]}")
    # print(f"pos_diff_proj:{pos_diff_proj.shape}")
    # print(f"pos_diff_orth:{pos_diff_orth.shape}")
    # print(f"pos_diff_proj_len:{pos_diff_proj_len.shape}")
    # print(f"pos_diff_orth_len:{pos_diff_orth_len.shape}")
    # print(f"pos_diff_to_end_pusher:{pos_diff_to_end_pusher.shape}")
    # print(f"pos_diff_l_mask:{pos_diff_l_mask.shape}")
    # print(f"pos_diff_w_mask:{pos_diff_w_mask.shape}")
    
    # reshape
    state = state_ori.transpose(1, 2).reshape(B, N, -1) # [B, N, n_his*2]
    attr = attr.transpose(1, 2).reshape(B, N, -1) # [B, N, n_his*2]
    # print(f"state: {state.shape}")
    return attr, state, edge_index