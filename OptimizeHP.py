import gpytorch
import torch
from torch.distributions import MultivariateNormal
from learning_state_dynamics import TARGET_POSE_OBSTACLES, BOX_SIZE, ResidualDynamicsModel, PushingController, obstacle_avoidance_pushing_cost_function
import numpy as np
from panda_pushing_env import *


#GAME PLAN
'''
Steps to TS-GP optimization:
1)generate dataset
sweep through range of hyperparameters:
noise_sigma val
lambda_value

2) train gp hyperparameters over the function:
cost function vs hyper parameters
^cost function will have to differ from pushing cost function... likely function of steps necessary and if goal has been reached

issue: current obstacle course can vary difficulty a lot

3)Run TS-GP algorithm with trained GP model
Sample points until we have enough to fit GP
Once we have enough, until we see small change in performance, Fit GP to observations, sample from GP fit, find min of sampled GP and add to observations
-> stonks

4)run CMA-ES algorithm
'''



pushing_multistep_residual_dynamics_model = ResidualDynamicsModel(3,3)
model_path = os.path.join('pushing_multi_step_residual_dynamics_model.pt')
pushing_multistep_residual_dynamics_model.load_state_dict(torch.load(model_path))


def execute(env, model, controller, state_0, num_steps_max = 20):
    state = state_0
    for i in range(num_steps_max):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        if done:
            break
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE
    return i, goal_distance, goal_reached

def execution_cost(i, goal_distance):
    # A far distance is 0.4, A pass can get something like .06
    cost = i + 10*goal_distance 
    return cost


def collect_data_random(env, model, controller, dataset_size = 100):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data = []
    env = PandaPushingEnv(visualizer=None, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
    for i in range (dataset_size):
        state_0 = env.reset()
        controller = PushingController(env, pushing_multistep_residual_dynamics_model,
                               obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
        data = {}
        data['hyperparameters'] = np.zeros(5, dtype = np.float32) #noise_sigma, lambda_value, x, y, theta
        data['cost'] = 0
        # Randomly Sample Hyperparameter values
        #SAMPLING STUFGFF
        #WEFASEDRHGERTSIHJNSE
        # Simulate using these hyperparameters
        controller.mppi.noise_sigma = data['hyperparameters'][0]
        controller.lambda_ = data['hyperparameters'][1]        
        steps, goal_distance, _ = execute(env, controller, state_0)
        # Add cost to data
        data['cost'] = execution_cost(steps, goal_distance)
    #   

    # ---
    return collected_data


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    total_Dataset = SingleStepDynamicsDataset(collected_data)
    train_set_size = int(0.8*len(total_Dataset))
    val_set_size = len(total_Dataset) - train_set_size

    train_set, val_set = random_split(total_Dataset, [train_set_size, val_set_size])

    train_loader = DataLoader(train_set, batch_size = batch_size)
    val_loader = DataLoader(val_set, batch_size = batch_size)
    # ---
    return train_loader, val_loader
