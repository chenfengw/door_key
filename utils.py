import os
import numpy as np
import gym
import gym_minigrid
import pickle
import matplotlib.pyplot as plt
import imageio
import random
import copy

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

IDX_TO_DIR = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}  # direction to array
DIR_TO_IDX = {val : key for key, val in IDX_TO_DIR.items()}

def step(env, action):
    '''
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    '''
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle
        }

    _, _, done, _ = env.step(actions[action])
    plot_env(env)
    return done

def generate_random_env(seed, task):
    ''' 
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    '''
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task)
    env.seed(seed)
    env.reset()
    return env

def load_env(path, load_random_env=False):
    '''
    Load Environments
    path: env path or random folder path
    ---------------------------------------------
    Returns:
        gym-environment, info
    '''
    if load_random_env:
        env_list = [os.path.join(path, env_file) for env_file in os.listdir(path)]
        path = random.choice(env_list)
    with open(path, 'rb') as f:
        env = pickle.load(f)
    
    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec,
        'door_pos': [],
        'door_open': []
        }
    
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i),
                          gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Door):
                info['door_pos'].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info['door_open'].append(True)
                else:
                    info['door_open'].append(False)
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])
            
    return env, info, path

def save_env(env, path):
    with open(path, 'wb') as f:
        pickle.dump(env, f)

def plot_env(env):
    '''
    Plot current environment
    ----------------------------------
    '''
    img = env.render('rgb_array', tile_size=32)
    plt.figure()
    plt.imshow(img)
    plt.show()

def draw_gif_from_seq(seq, env, save_name="test"):
    '''
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]
    
    env:
        The doorkey environment
    '''
    path = f"./gif_test/{save_name}.gif"
    with imageio.get_writer(path, mode='I', duration=0.8) as writer:
        img = env.render('rgb_array', tile_size=32)
        writer.append_data(img)
        for act in seq:
            print(f"this action is : {act}")
            step(env, act)
            img = env.render('rgb_array', tile_size=32)
            writer.append_data(img)
    print('GIF is written to {}'.format(path))
    
def is_cell(cell_pos, cell_type , env):
    type_map = {"Key": gym_minigrid.minigrid.Key,
                "Goal": gym_minigrid.minigrid.Goal,
                "Door": gym_minigrid.minigrid.Door,
                "Wall": gym_minigrid.minigrid.Wall}
    return isinstance(env.grid.get(*cell_pos), type_map[cell_type])

def get_door_index(cell_pos, all_doors):
    """
    Given position of a door and list of doors, 
    return the index of the odor
    """
    return [np.array_equal(cell_pos, d) for d in all_doors].index(True)

def hash_state(state):
    """
    Turn a state dictionary to tuples.
    """
    state_new = copy.deepcopy(state)
    for key, val in state_new.items():
        if isinstance(val, np.ndarray): 
            state_new[key] = tuple(val)
        if isinstance(val, list):
            # convert list of np array to tuples
            if all([isinstance(item, np.ndarray) for item in val]):
                val_new = [tuple(item) for item in val] 
                state_new[key] = tuple(val_new)
            # convert list of bool to tuples
            elif all([isinstance(item, bool) for item in val]):
                state_new[key] = tuple(val)

    return tuple(state_new.items())

def get_initial_state(info):
    state = {}
    state['agent_pos'] = info['init_agent_pos']
    state['agent_dir'] = tuple(info['init_agent_dir'])
    state['agent_carry'] = False
    state['door_open'] = info['door_open']
    state['door_pos'] = info['door_pos']
    state['key_pos'] = info['key_pos']
    state['goal_pos'] = info['goal_pos']
    return state

def is_goal(state, goal_pos):
    return np.array_equal(state["agent_pos"], goal_pos)