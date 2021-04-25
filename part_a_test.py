# %%
import utils
import numpy as np
import itertools
import copy
import importlib
import itertools
importlib.reload(utils)
import dp

# %%
env_folder = './envs/random_envs'
env, info = utils.load_env('./envs/doorkey-5x5-normal.env')
utils.plot_env(env)
# %%
def generate_state_space(env, info):
    grid_coordinates = np.arange(env.height)
    all_agent_pos = [np.array(item) for item in itertools.product(grid_coordinates, repeat=2)]
    states_combo = {"agent_pos": all_agent_pos, 
                    "agent_dir": [(0,-1), (1,0), (0,1), (-1,0)], 
                    "agent_carry": [True, False],
                    "door_pos":info["door_pos"],
                    "door_open":[True, False],
                    "key_pos": info["key_pos"],
                    "goal_pos":info["goal_pos"]
                    }
    keys, values = zip(*states_combo.items())
    state_space_all = [dict(zip(keys, v)) for v in itertools.product(*values)]
    state_space = []

    for state in state_space_all:
        if not utils.is_cell(state["agent_pos"], "Wall", env):
            if not isinstance(state["door_pos"], list):
                state["door_pos"] = [state["door_pos"]]
            if not isinstance(state["door_open"], list):
                state["door_open"] = [state["door_open"]]
            state_space.append(state)
    
    # compute state to index table
    n_states = len(state_space)
    state_to_idx = {utils.hash_state(state): idx for state, idx in zip(state_space, range(n_states))}
    return state_space, state_to_idx

# %%
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

# %%
# state = {}
# state["agent_pos"] = np.array([3,1])
# state["agent_dir"] = idx_to_dir[3]
# state["agent_carry"] = True
# state["door_pos"] = info["door_pos"]
# state["door_open"] = info["door_open"]
# state["key_pos"] = info["key_pos"]
# state["goal_pos"] = info["goal_pos"]

# env.agent_pos = state["agent_pos"]
# env.agent_dir = 3
# env.carrying = True
# print(state)
# utils.plot_env(env)
# # %%
# next_state = get_next_state(state, MF, env)
# print(next_state)

# %%
state_space, state_to_idx = generate_state_space(env, info)

# %%
def DP(state_space, state_to_idx, control_space, get_next_state, stage_cost, terminal_cost, env):
    T = len(state_space) - 1
    V = np.zeros([T+1, len(state_space)])
    pi = np.zeros_like(V)
    
    # initialize the value functions
    for idx, state in enumerate(state_space):
        V[T,idx] = terminal_cost(state)
    
    for t in reversed(range(T)):
        Q = np.zeros([len(state_space), len(control_space)])

        for state_idx, state in enumerate(state_space):
            for ctrl_idx, control in enumerate(control_space):
                next_state =  get_next_state(state, control, env)
                # print(next_state)
                next_state_idx = state_to_idx[utils.hash_state(next_state)]
                # compute Q function
                Q[state_idx, ctrl_idx] = stage_cost(control) + V[t+1, next_state_idx]
            
            # compute cost and policy
            if V[t+1, state_idx] == float("inf"):
                V[t, state_idx] = min(Q[state_idx,:]) # compute min Q[state_idx,:]
            else:
                V[t, state_idx] = V[t+1, state_idx]
            pi[t, state_idx] = control_space[Q[state_idx,:].argmin()] # argmin Q[state_idx,:]

        # early stopping
        if np.array_equal(V[t,:], V[t+1,:]):
            print("early stopping")
            return V[t+1:], pi[t+1:]
    return V, pi

# %%
control_space = [MF, TL, TR, PK, UD]

V, pi = DP(state_space, state_to_idx, control_space, dp.get_next_state, dp.step_cost, dp.terminal_cost, env)
# %%
