import gpytorch
import torch
from torch.distributions import MultivariateNormal
from learning_state_dynamics import TARGET_POSE_OBSTACLES, BOX_SIZE, ResidualDynamicsModel, PushingController, obstacle_avoidance_pushing_cost_function
import numpy as np
from panda_pushing_env import *
from tqdm import tqdm

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



# pushing_multistep_residual_dynamics_model = ResidualDynamicsModel(3,3)
# model_path = os.path.join('pushing_multi_step_residual_dynamics_model.pt')
# pushing_multistep_residual_dynamics_model.load_state_dict(torch.load(model_path))




def execute(env, controller, state_0, num_steps_max = 20):
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

def collect_data_GP(env, controller, dataset_size = 20):
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
    # --- Your code here
    collected_data = []
    pbar = tqdm(range(dataset_size))

    for i in pbar:
        pbar.set_description(f'Iteration: {i:.0f}')
        state_0 = env.reset()
        data = {}
        data['hyperparameters'] = np.zeros(4, dtype = np.float32) #noise_sigma, lambda_value, x, y, theta
        data['cost'] = 0
        # Randomly Sample Hyperparameter values
        data['hyperparameters'][0] = np.random.uniform(0, 2)
        data['hyperparameters'][1] = np.random.uniform(0, 0.015)
        data['hyperparameters'][2] = np.random.uniform(0, 2)
        data['hyperparameters'][3] = np.random.uniform(0, 1)
        #Should we also consider changing horizon?
        # Simulate using these hyperparameters
        controller.mppi.noise_sigma = data['hyperparameters'][0]
        controller.lambda_ = data['hyperparameters'][1]
        controller.linear_weight = data['hyperparameters'][2]
        controller.theta_weight = data['hyperparameters'][3]
        steps, goal_distance, _ = execute(env, controller, state_0)
        # Add cost to data
        data['cost'] = execution_cost(steps, goal_distance)
        collected_data.append(data)
    #   

    # ---
    return collected_data


class RBF_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        # --- Your code here
        self.mean_module = gpytorch.means.ZeroMean(torch.Size([5]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([5]), ard_num_dims = 5),
            batch_shape=torch.Size([5])
        )
        # ---
    def forward(self, x):
        """
        Args:
            x: torch.tensor of shape (B, 5) concatenated state and action

        Returns: gpytorch.distributions.MultitaskMultivariateNormal - Gaussian prediction for next state

        """
        mean_x = self.mean_module.forward(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


def train_gp_hyperparams(model, likelihood, hyperparameters, cost, lr):
    """
        Function which optimizes the GP Kernel & likelihood hyperparameters
    Args:
        model: gpytorch.model.ExactGP model
        likelihood: gpytorch likelihood
        hyperparameters: (N, 5) torch.tensor of hyperpameters
        train_next_states: (N,) torch.tensor of cost due to hyperpameters
        lr: Learning rate

    """
    # --- Your code here
    training_iter = 115
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(hyperparameters)
        # Calc loss and backprop gradients
        loss = -mll(output, cost)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
    # ---