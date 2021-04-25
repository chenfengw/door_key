# %%
import utils
import numpy as np
import itertools
import copy
import gym_minigrid
# %%
task = 'MiniGrid-DoorKey-5x5-v0'
env = utils.generate_random_env(-1, task)
# %%
utils.plot_env(env)
# %%

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def example_use_of_gym_env():
    '''
    The Coordinate System:
        (0,0): Top Left Corner
        (x,y): x-th column and y-th row
    '''
    
    print('<========== Example Usages ===========> ')
    env_path = './envs/example-8x8.env'
    # env, info = load_env(env_path) # load an environment
    
    env, info = utils.load_env('./envs/doorkey-8x8-shortcut.env')
    print('<Environment Info>\n')
    print(info) # Map size
                # agent initial position & direction, 
                # key position, door position, goal position
    print('<================>\n')            
    
    # Visualize the environment
    utils.plot_env(env) 
    
    
    # Get the agent position
    agent_pos = env.agent_pos
    
    # Get the agent direction
    agent_dir = env.dir_vec # or env.agent_dir
    
    # Get the cell in front of the agent
    front_cell = front_pos # == agent_pos + agent_dir
    
    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3) # NoneType, Wall, Key, Goal
    
    # Get the door status
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    is_open = door.is_open
    is_locked = door.is_locked
    
    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None
    
    # Take actions
    cost, done = utils.step(env, MF) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Moving Forward Costs: {}'.format(cost))
    cost, done = utils.step(env, TL) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Turning Left Costs: {}'.format(cost))
    cost, done = utils.step(env, TR) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Turning Right Costs: {}'.format(cost))
    cost, done = utils.step(env, PK) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Picking Up Key Costs: {}'.format(cost))
    cost, done = utils.step(env, UD) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Unlocking Door Costs: {}'.format(cost))   
    
    # Determine whether we stepped into the goal
    if done:
        print("Reached Goal")
    
    # The number of steps so far
    print('Step Count: {}'.format(env.step_count))

    return env
# %%
env = example_use_of_gym_env()
# %%
env_folder = './envs/random_envs'
env, info = utils.load_env('./envs/doorkey-8x8-shortcut.env')
utils.plot_env(env)
# %%
def get_state_space(env, info):
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
                if not isinstance(env.grid.get(*an_agent_pos), gym_minigrid.minigrid.Wall): 
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
    if env.grid.get(*front_pos) is None or front_pos == current_state["goal_pos"]:
        next_state["agent_pos"] = front_pos
        return next_state
    elif current_state["agent_carry"] and \
         isinstance(env.grid.get(*front_pos), 
                    gym_minigrid.minigrid.Key):
        next_state["agent_pos"] = front_pos
        return next_state
    elif current_state["door_open"] and \
        isinstance(env.grid.get(*front_pos), 
                   gym_minigrid.minigrid.Door):
        next_state["agent_pos"] = front_pos
        return next_state
    else:
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
    if not current_state["agent_carry"] and \
        isinstance(env.grid.get(*front_pos), 
                   gym_minigrid.minigrid.Key):
        next_state["agent_carry"] = True
    else:
        return current_state

def try_unlock_door(current_state, action, env):
    next_state = copy.deepcopy(current_state)
    front_pos = current_state["agent_pos"] + current_state["agent_dir"]

    if isinstance(env.grid.get(*front_pos), 
                  gym_minigrid.minigrid.Door) and \
       current_state["agent_carry"] and \
       not current_state["door_open"]:
        next_state["door_open"] = True
    else:
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
state["agent_pos"] = env.agent_pos
state["agent_dir"] = env.agent_
state["agent_carry"] = is_agent_carry
state["door_pos"] = info["door_pos"]
state["door_open"] = info["door_open"]
state["key_pos"] = info["key_pos"]
state["goal_pos"] = info["goal_pos"]