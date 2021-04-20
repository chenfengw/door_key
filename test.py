# %%
import utils
# %%
task = 'MiniGrid-DoorKey-5x5-v0'
env = utils.generate_random_env(-1, task)
# %%
utils.plot_env(env)
# %%
