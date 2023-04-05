import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm
from OptimizeHP import collect_data_GP
from learning_state_dynamics import *
    from panda_pushing_env import *
#### Loading Memory Profiler Package

pushing_multistep_residual_dynamics_model = ResidualDynamicsModel(3,3)
model_path = os.path.join('pushing_multi_step_residual_dynamics_model.pt')
pushing_multistep_residual_dynamics_model.load_state_dict(torch.load(model_path))

env = PandaPushingEnv(visualizer=None, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
env.reset()

controller = PushingController(env, pushing_multistep_residual_dynamics_model,
                        obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
collected_data = collect_data_GP(env, controller)
np.save(os.path.join('collected_data0.npy'), collected_data)