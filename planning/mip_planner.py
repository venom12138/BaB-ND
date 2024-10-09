import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from model.mlp import MLP


class MIP_Planner(object):
    def __init__(self, model: MLP):
        self.model = model
        self.mip_config = model.config["planning"]["mip"]
        self.time_limit = model.config["planning"]["timeout"]

        self.params = list(self.model.model.parameters())
        self.params = [p.data.cpu().numpy() for p in self.params]

        # prune weights
        if self.mip_config["prune_weights"]:
            for param in self.params:
                if len(param.shape) == 2:  # If are w's
                    for i in range(param.shape[0]):
                        for j in range(param.shape[1]):
                            if abs(param[i, j]) < self.mip_config["prune_weights_threshold"]:
                                param[i, j] = 0.0
                else:
                    pass

        # verify bounds
        if self.mip_config["verify_bounds"]:
            self.model.verify_bounds()

    def trajectory_optimization(
        self,
        state_cur: np.ndarray,  # current state, [n_his, state_dim]
        cur_pusher_pos: np.ndarray,  # current pusher position, [2]
        obs_goal: np.ndarray,  # goal, [state_dim]
        act_seq: np.ndarray,  # initial action sequence, [-1, action_dim]
        bound: float,  # bound of action
        n_look_ahead: int,  # number of look ahead steps/horizon
        only_final_cost: bool = False,  # whether to only use the final cost
    ):
        state_goal = obs_goal
        N = n_look_ahead

        lb_activation = self.model.lb_activation
        ub_activation = self.model.ub_activation

        n_his = state_cur.shape[0]
        state_dim = state_cur.shape[-1]
        action_dim = act_seq.shape[-1]

        act_his = act_seq[: n_his - 1]
        # import pdb; pdb.set_trace()
        # n_relu = len(self.model.mask)
        # # mask: shape is [#Relu layers, 1, #output units at that Relu layer]
        # mask = self.model.mask
        # mask = [p.data.cpu().numpy() for p in mask]
        
        # # 0804 ANONYMOUS: only support relu now
        model_architecture = self.model.config["train"]["architecture"]
        # import pdb; pdb.set_trace()
        n_relu = len(model_architecture)
        mask = [np.ones((1, int(model_architecture[i]))) for i in range(n_relu)]

        ### build up the MIP
        m = gp.Model("mip")
        m.Params.TimeLimit = self.time_limit
        m.Params.LogToConsole = 0
        m.Params.OutputFlag = 0
        m.Params.MIPGap = 1e-6  # default 1e-4

        # state variable
        x = m.addMVar(shape=(n_his + N) * state_dim, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

        # action variable (planned)
        a0 = m.addMVar(shape=(N * action_dim), vtype=GRB.CONTINUOUS, lb=-bound, ub=bound, name="a0")

        # the entire action sequence including history
        a = m.addMVar(shape=(n_his + N - 1) * action_dim, vtype=GRB.CONTINUOUS, lb=-bound, ub=bound, name="a")

        # set up the variable dictionary
        varDict = {"x": x, "a0": a0, "a": a}  # 'a1': a1, 'a': a}

        # add constraint between action variables
        m.addConstr(a[(n_his - 1) * action_dim :: 1] == a0)

        # binary variables, one for each relu
        z = []

        # constain the initial condition
        m.addConstr(x[: n_his * state_dim] == state_cur.reshape(-1))
        m.addConstr(a[: (n_his - 1) * action_dim] == act_his.reshape(-1))

        for t in range(N):  # for the horizon we are planning over
            z.append([])

            # current state and action
            x_cur = x[t * state_dim : (t + n_his) * state_dim]
            a_cur = a[t * action_dim : (t + n_his) * action_dim]

            # for variable naming: next_s = relu(y) = relu(w * s + b)
            name = "s_%d_%d" % (t, 0)
            s = m.addMVar(
                shape=x_cur.shape[0] + a_cur.shape[0],
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                name=name,
            )
            varDict[name] = s  # input

            m.addConstr(s[: x_cur.shape[0]] == x_cur)
            m.addConstr(s[x_cur.shape[0] :] == a_cur)

            # passing through the neural network layer by layer
            for idx_layer in range(n_relu):
                mask_pos = mask[idx_layer][0] == 2
                mask_neg = mask[idx_layer][0] == 0
                mask_others = mask[idx_layer][0] == 1

                # activation before relu
                ss = self.params[idx_layer * 2] @ s + self.params[idx_layer * 2 + 1]
                n_neuron = ss.shape[0]

                # print("w: ", self.params[idx_layer * 2])
                # print("b: ", self.params[idx_layer * 2 + 1])

                # activation after relu and mask
                name = "y_%d_%d" % (t, idx_layer + 1)
                y = m.addMVar(shape=n_neuron, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=name)
                varDict[name] = y  # output of linear, pre-relu

                m.addConstr(y == ss)

                # represents output of this layer
                name = "s_%d_%d" % (t, idx_layer + 1)
                s = m.addMVar(shape=n_neuron, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=name)
                varDict[name] = s  # output of regular relu, to be post-relu

                # positive/identity neuron
                m.addConstr(s[mask_pos] == y[mask_pos])
                # negative neuron
                m.addConstr(s[mask_neg] == 0.0)

                # the binary variable
                name = "z_%d_%d" % (t, idx_layer + 1)
                z[t].append(m.addMVar(shape=n_neuron, vtype=GRB.BINARY, name=name))
                varDict[name] = z[t][-1]  # regular relu

                for idx_neuron in range(n_neuron):
                    if mask_others[idx_neuron] == 1:
                        # regular relu
                        lb_cur = (
                            lb_activation[idx_layer][idx_neuron] - self.mip_config["bounds_extension"]
                        )
                        ub_cur = (
                            ub_activation[idx_layer][idx_neuron] + self.mip_config["bounds_extension"]
                        )
                        z_cur = z[t][idx_layer][idx_neuron]
                        m.addConstr(s[idx_neuron] <= y[idx_neuron] - lb_cur * (1 - z_cur))
                        m.addConstr(s[idx_neuron] >= y[idx_neuron])
                        m.addConstr(s[idx_neuron] <= ub_cur * z_cur)
                        m.addConstr(s[idx_neuron] >= 0.0)

            # calculate the residual by passing through last layer (without relu)
            s = self.params[n_relu * 2] @ s + self.params[n_relu * 2 + 1]

            # calculate the next state
            xx = x[(t + n_his - 1) * state_dim : (t + n_his) * state_dim]  # curr state
            # import pdb; pdb.set_trace()
            s = s + xx
            for idx in range(0, state_dim):
                s[idx] -= a_cur[idx % action_dim]

            m.addConstr(s == x[(t + n_his) * state_dim : (t + n_his + 1) * state_dim])

        name = "residual"
        residual = m.addMVar(shape=1, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=name)
        varDict[name] = residual
        total_residual = 0

        pusher_x = cur_pusher_pos[0]
        pusher_y = cur_pusher_pos[1]

        step_weight_list = (np.arange(1, N + 1) / N).tolist()

        for t in range(N):
            pusher_x += a0[t * action_dim]
            pusher_y += a0[t * action_dim + 1]
            rel_pos = x[(n_his + t) * state_dim : (n_his + t + 1) * state_dim]
            x_residual = rel_pos[0:state_dim:2] + pusher_x - state_goal[0:state_dim:2]
            y_residual = rel_pos[1:state_dim:2] + pusher_y - state_goal[1:state_dim:2]
            step_residual = step_weight_list[t] * (x_residual @ x_residual + y_residual @ y_residual)
            total_residual += step_residual
        if only_final_cost:
            total_residual = x_residual @ x_residual + y_residual @ y_residual
        m.addConstr(residual == total_residual)
        cost = residual

        m.setObjective(cost, GRB.MINIMIZE)

        start_time = time.perf_counter()
        m.update()
        m.optimize()
        end_time = time.perf_counter()
        solve_time = end_time - start_time
        print(f"solve time: {solve_time}")
        if m.Status == GRB.TIME_LIMIT:
            print(f"Time limit reached, {m.SolCount} solutions are found, but may be suboptimal.")
        try:
            a_sol = a.X.reshape(n_his + N - 1, action_dim)
            action_seq_future = a_sol[n_his - 1 :]

            x_sol = x.X.reshape(n_his + N, state_dim)
            obs_seq_best = x_sol[n_his:]

            reward_best = m.objVal

            varSol = {}
            for key in varDict.keys():
                varSol[key] = varDict[key].X
            # print("SOLUTION FOUND!")
            return {
                "action_sequence": action_seq_future,  # [n_roll, action_dim]
                "observation_sequence": obs_seq_best,  # [n_roll, obs_dim]
                "objective": reward_best,
                "varSol": varSol,
                "solve_status": m.Status,
                "solve_time": solve_time,
            }
        except:
            return {"solve_status": m.Status, "solve_time": solve_time}
