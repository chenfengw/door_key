# %%
import os
import gym
import numpy as np
import itertools
import importlib

import utils
import dp

importlib.reload(utils)
importlib.reload(dp)
# %%
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def generate_state_space(env, info):
    grid_coordinates = np.arange(env.height)
    all_agent_pos = [np.array(item) for item in itertools.product(grid_coordinates, repeat=2)]
    states_combo = {"agent_pos": all_agent_pos, 
                    "agent_dir": [(0,-1), (1,0), (0,1), (-1,0)], 
                    "agent_carry": [True, False]}
    
    if len(info["door_pos"]) == 1: # if 1 door
        states_combo["door_open"] = [[True], [False]]
    
    elif len(info["door_pos"]) == 2: # if 2 door
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

def doorkey_problem(env, info):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    state_space, state_to_idx = generate_state_space(env, info)
    control_space = [MF, TL, TR, PK, UD]
    V, pi = dp.DP(state_space, state_to_idx, control_space, dp.get_next_state, dp.step_cost, dp.terminal_cost, env)
    
    # get optimal sequence
    state = utils.get_initial_state(info)
    optim_act_seq = []
    t = 0
    while not utils.is_goal(state, info["goal_pos"]):
        state_idx = state_to_idx[utils.hash_state(state)]
        optimal_control = pi[t, state_idx]
        optim_act_seq.append(optimal_control)
        
        # get next state
        state = dp.get_next_state(state, optimal_control, env)
        t += 1

    return optim_act_seq



def partA():
    env_path = './envs/doorkey-6x6-direct.env'
    env, info, _ = utils.load_env(env_path) # load an environment
    seq = doorkey_problem(env, info) # find the optimal action sequence
    fig_name = os.path.basename(env_path).split(".")[0]
    utils.draw_gif_from_seq(seq, utils.load_env(env_path)[0], save_name=fig_name) # draw a GIF & save
    return seq, env

def partB():
    env_folder = './envs/random_envs'
    env, info, env_path =  utils.load_env(env_folder, load_random_env=True)
    seq = doorkey_problem(env, info) # find the optimal action sequence
    fig_name = os.path.basename(env_path).split(".")[0]
    utils.draw_gif_from_seq(seq, env, save_name=fig_name) # draw a GIF & save
# if __name__ == '__main__':
#     #example_use_of_gym_env()
#     partA()
#     #partB()

# %%
seq, env = partA()
# %%
partB()
# %%
