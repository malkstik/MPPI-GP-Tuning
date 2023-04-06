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
import gpytorch
#### Loading Memory Profiler Package

# pushing_multistep_residual_dynamics_model = ResidualDynamicsModel(3,3)
# model_path = os.path.join('pushing_multi_step_residual_dynamics_model.pt')
# pushing_multistep_residual_dynamics_model.load_state_dict(torch.load(model_path))

# env = PandaPushingEnv(visualizer=None, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
# env.reset()

# controller = PushingController(env, pushing_multistep_residual_dynamics_model,
#                         obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
# collected_data = collect_data_GP(env, controller)
# np.save(os.path.join('collected_data0.npy'), collected_data)


class RBF_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        # --- Your code here
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        # ---
    def forward(self, x):
        """
        Args:
            x: torch.tensor of shape (B, 4) containing all hyperparameters

        Returns: gpytorch.distributions.MultitaskMultivariateNormal - Gaussian prediction for next state

        """
        mean_x = self.mean_module.forward(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    
    def predict(self, x):
        pred = self.likelihood(self.forward(x))
        pred_mu = pred.mean
        pred_sigma = torch.diag_embed(pred.stddev ** 2)

        return pred_mu, pred_sigma


# train_x = torch.linspace(0,1,100)
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
# # We will use the simplest form of GP model, exact inference
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # initialize likelihood and model
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood)


x1 = torch.linspace(0,1,100)
x2 = torch.linspace(0,1,100)
# train_x = x1

train_y = (x1**2 + torch.cos(x1))
#train_y = (x1**2 + torch.cos(x2)).unsqueeze(1)
# print(train_y.shape)

train_x = torch.stack((x1,x2), -1)
#train_y = train_y.reshape((-1, train_y.shape[0]))
print(train_x.shape)
print(train_y.shape)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = RBF_GP(train_x, train_y, likelihood)

likelihood.eval()
model.eval()

qp  = torch.tensor([[0.5, 0.1]])
print(qp.shape)
pred = likelihood(model(qp))
print(pred.mean)


