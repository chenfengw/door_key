# %%
import utils
import numpy as np
import itertools
import copy
import importlib
import itertools
import dp
importlib.reload(utils)
importlib.reload(dp)


# %%
env_folder = './envs/random_envs'
env, info = utils.load_env('./envs/doorkey-5x5-normal.env')
utils.plot_env(env)

random_env, random_info = utils.load_env(env_folder, load_random_env=True)
# %%
def generate_state_space(env, info):
    grid_coordinates = np.arange(env.height)
    all_agent_pos = [np.array(item) for item in itertools.product(grid_coordinates, repeat=2)]
    states_combo = {"agent_pos": all_agent_pos, 
                    "agent_dir": [(0,-1), (1,0), (0,1), (-1,0)], 
                    "agent_carry": [True, False]
                    }
    if len(info["door_pos"]) == 1:
        states_combo["door_open"] = [[True], [False]]
    elif len(info["door_pos"]) == 2:
        states_combo["door_open"] = [[False,False],[False,True],[True,False],[True,True]]
    
    keys, values = zip(*states_combo.items())
    state_space_all = [dict(zip(keys, v)) for v in itertools.product(*values)]
    state_space = []

    for state in state_space_all:
        if not utils.is_cell(state["agent_pos"], "Wall", env):
            state["door_pos"] = info["door_pos"]
            state["key_pos"] = info["key_pos"] # 1 key
            state["goal_pos"] = info["goal_pos"] # 1 goal
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
state_space1 , state_to_idx1 = generate_state_space(random_env, random_info)
state_space2,  state_to_idx2 = generate_state_space(env, info)
control_space = [MF, TL, TR, PK, UD]

init1 = utils.get_initial_state(random_info)
init2 = utils.get_initial_state(info)
# %%
V, pi = dp.DP(state_space, state_to_idx, control_space, dp.get_next_state, dp.step_cost, dp.terminal_cost, env)
# %%
