# %%
import utils
import numpy as np
import itertools
import copy
import importlib

importlib.reload(utils)
from utils import is_cell, get_door_index, hash_state

# %%
env_folder = './envs/random_envs'
env, info = utils.load_env('./envs/doorkey-8x8-shortcut.env')
utils.plot_env(env)
# %%
def generate_state_space(env, info):
    grid_coordinates = np.arange(env.height)
    all_agent_pos = [np.array(item) for item in itertools.product(grid_coordinates, repeat=2)]
    states_combo = {"agent_pos": all_agent_pos, 
                    "agent_dir": [(0,-1), (1,0), (0,1), (-1,0)], 
                    "agent_carry": [True, False],
                    "door_pos":[[0,1], [0,2]],
                    "door_open":[True, False],
                    "key_pos":[[0,3], [0,4]],
                    "goal_pos":[[0,1],[0,2]]}
    state_space = []
    for an_agent_pos in states_combo["agent_pos"]:
        for an_agent_dir in states_combo["agent_dir"]:
            for is_agent_carry in states_combo["agent_carry"]:
                if not is_cell(an_agent_pos, "Wall", env): 
                    state = {}
                    state["agent_pos"] = an_agent_pos
                    state["agent_dir"] = an_agent_dir
                    state["agent_carry"] = is_agent_carry
                    state["door_pos"] = info["door_pos"]
                    state["door_open"] = info["door_open"]
                    state["key_pos"] = info["key_pos"]
                    state["goal_pos"] = info["goal_pos"]
                    state_space.append(state)
    
    return state_space

# %%
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door
idx_to_dir = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}
dir_to_idx = {val : key for key, val in idx_to_dir.items()}

def try_move_forward(current_state, action, env):
    next_state = copy.deepcopy(current_state)
    front_pos = current_state["agent_pos"] + current_state["agent_dir"]
    # print(front_pos)
    # np.array_equal(front_pos, current_state["goal_pos"])
    if env.grid.get(*front_pos) is None or is_cell(front_pos,"Goal",env):
        next_state["agent_pos"] = front_pos
        return next_state
    if current_state["agent_carry"] and is_cell(front_pos,"Key",env):
        next_state["agent_pos"] = front_pos
        return next_state
    if is_cell(front_pos, "Door", env):
        door_idx = get_door_index(front_pos, current_state["door_pos"])
        if current_state["door_open"][door_idx]:
            next_state["agent_pos"] = front_pos
            return next_state
   
    return current_state

def try_turn_left_right(current_state, action, env):
    next_state = copy.deepcopy(current_state)
    idx = dir_to_idx[current_state["agent_dir"]]
    if action == TL:
        next_idx = (idx - 1) % 4  # 0 - 3
    elif action == TR:
        next_idx = (idx + 1) % 4
    next_state["agent_dir"] = idx_to_dir[next_idx] # (0,-1), etc. tuple
    return next_state

def try_pickup_key(current_state, action, env):
    next_state = copy.deepcopy(current_state)
    front_pos = current_state["agent_pos"] + current_state["agent_dir"]
    if not current_state["agent_carry"] and is_cell(front_pos,"Key",env):
        next_state["agent_carry"] = True
        return next_state
    else:
        return current_state

def try_unlock_door(current_state, action, env):
    next_state = copy.deepcopy(current_state)
    front_pos = current_state["agent_pos"] + current_state["agent_dir"]

    if current_state["agent_carry"] and is_cell(front_pos,"Door",env):
        door_idx = get_door_index(front_pos, current_state["door_pos"])
        if current_state["door_open"][door_idx] == False:
            next_state["door_open"][door_idx] = True
            return next_state
    return current_state

def get_next_state(current_state, action, env):
    if action == MF:
        return try_move_forward(current_state, action, env)
    elif action == TL or action == TR:
        return try_turn_left_right(current_state, action, env)
    elif action == PK:
        return try_pickup_key(current_state, action, env)
    elif action == UD:
        return try_unlock_door(current_state, action, env)
# %%
state = {}
state["agent_pos"] = np.array([3,1])
state["agent_dir"] = idx_to_dir[3]
state["agent_carry"] = True
state["door_pos"] = info["door_pos"]
state["door_open"] = info["door_open"]
state["key_pos"] = info["key_pos"]
state["goal_pos"] = info["goal_pos"]

env.agent_pos = state["agent_pos"]
env.agent_dir = 3
env.carrying = True
print(state)
utils.plot_env(env)
# %%
next_state = get_next_state(state, MF, env)
print(next_state)

# %%
state_space = generate_state_space(env, info)
n_states = len(state_space)
state_to_idx = {hash_state(state): idx for state, idx in zip(state_space, range(n_states))}
# %%
def DP(state_space, control_space, get_next_state, T, stage_cost, terminal_cost):
    value = np.zeros(T+1, len(state_space))
    policy = np.zeros(len(state_space), len(control_space))

    # initialize the value functions

