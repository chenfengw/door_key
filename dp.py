import copy
from utils import *

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
    idx = DIR_TO_IDX[current_state["agent_dir"]]
    if action == TL:
        next_idx = (idx - 1) % 4  # 0 - 3
    elif action == TR:
        next_idx = (idx + 1) % 4
    next_state["agent_dir"] = IDX_TO_DIR[next_idx] # (0,-1), etc. tuple
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
    """
    Motion model x_t+1 = f(x, u)
    """
    if action == MF:
        return try_move_forward(current_state, action, env)
    elif action == TL or action == TR:
        return try_turn_left_right(current_state, action, env)
    elif action == PK:
        return try_pickup_key(current_state, action, env)
    elif action == UD:
        return try_unlock_door(current_state, action, env)

def step_cost(action):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    return 1 # the cost of action

def terminal_cost(state):
    if np.array_equal(state["agent_pos"], state["goal_pos"]):
        return 0
    return float('inf')