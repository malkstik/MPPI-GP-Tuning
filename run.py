import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cma

from torch.distributions.multivariate_normal import MultivariateNormal
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm
from learning_state_dynamics import collect_data_random
from learning_state_dynamics import ResidualDynamicsModel, PushingController, obstacle_avoidance_pushing_cost_function
from panda_pushing_env import PandaPushingEnv
from visualizers import *
from OptimizeHP import *

pushing_multistep_residual_dynamics_model = ResidualDynamicsModel(3,3)
pushing_multistep_residual_dynamics_model.eval()
model_path = os.path.join('pushing_multi_step_residual_dynamics_model.pt')
pushing_multistep_residual_dynamics_model.load_state_dict(torch.load(model_path))

CMA_ENV = PandaPushingEnv(visualizer=None, render_non_push_motions=False,  
                    include_obstacle=True, camera_heigh=800, camera_width=800, 
                    render_every_n_steps=5)
state_0 = CMA_ENV.reset()
state = state_0
CMA_CONTROLLER = PushingController(CMA_ENV, pushing_multistep_residual_dynamics_model,
                            obstacle_avoidance_pushing_cost_function, 
                            num_samples=1000, horizon=30)

data_types = ['collected_data_OBS_1.npy',
            'collected_data_OBS_2.npy',
            'collected_data_OBS_3.npy']

GP_models= ['RBF_GP_model_OBS_1.pth',
            'RBF_GP_model_OBS_2.pth',
            'RBF_GP_model_OBS_3.pth']

TS_results = [['HP_OBS_1.pt', 'cost_OBS_1.pt'],
              ['HP_OBS_2.pt', 'cost_OBS_2.pt'],
              ['HP_OBS_3.pt', 'cost_OBS_3.pt']]

def check_env(save_gif = False):
    fig = plt.figure(figsize=(8,8))
    visualizer = GIFVisualizer()
    # Initialize the simulation environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True,
                        camera_heigh=800, camera_width=800, render_every_n_steps=5, obsInit = OBS_INIT)
    env.reset()
    # Perform 1 random action:
    for i in tqdm(range(1)):
        action_i = env.action_space.sample()
        state, reward, done, info = env.step(action_i)
        if done:
            break
    Image(filename=visualizer.get_gif())
    plt.close(fig)

def collect_data(obsInit):
    env = PandaPushingEnv(visualizer=None, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800,
                        camera_width=800, render_every_n_steps=5, obsInit = obsInit)
    env.reset()
    controller = PushingController(env, pushing_multistep_residual_dynamics_model,
                            obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=30)
    collected_data = collect_data_GP(env, controller)
    filename = "collected_data_OBS_" + str(obsInit) + ".npy"
    np.save(os.path.join(filename), collected_data)   

    return collected_data

def load_data(obsInit):
    filename = "collected_data_OBS_" + str(obsInit) + ".npy"
    data = np.load(os.path.join(filename), allow_pickle=True)
    data = data.reshape(-1)
    collected_data = {}

    train_x = torch.from_numpy(np.stack([d['hyperparameters'] for d in data], axis=0)).type(torch.float64)
    train_y = torch.from_numpy(np.stack([d['cost'] for d in data], axis=0)).type(torch.float64)    

    return train_x, train_y

def train_gp(train_x, train_y, obsInit):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()


    model = RBF_GP(train_x, train_y, likelihood)
    model.eval()
    likelihood.eval()

    train_gp_hyperparams(model, likelihood, train_x, train_y, mute = True)

    filename = "RBF_GP_model_OBS_" + str(obsInit) + ".pth"
    save_path = os.path.join(filename)
    torch.save(model.state_dict(), save_path)    
    return model.state_dict()

def run_TS(train_x, train_y, obsInit):
    filename = "RBF_GP_model_OBS_" + str(obsInit) + ".pth"
    GP_state_dict = torch.load(filename)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()    
    constraints = torch.tensor([[0, 10],
                                [0, 0.015],
                                [0, 5],
                                [0, 5],
                                [0, 5]])

    TS = ThompsonSamplingGP(GP_state_dict, likelihood, constraints, pushing_multistep_residual_dynamics_model, 
                            train_x, train_y, prior = None, obsInit = obsInit)
    optimum_hp, optimum_cost = TS.getOptimalParameters()

    TS_HP = TS.X
    TS_cost = TS.y
    HP_filename = "HP_OBS_" + obsInit + "1.pt"
    cost_filename = "cost_OBS_" + obsInit + "1.pt"
    torch.save(TS_HP, os.path.join(HP_filename))
    torch.save(TS_cost, os.path.join(cost_filename))

    return optimum_hp, optimum_cost

def CMA_evaluate(hyperparameters):
    #Change controller hyperparameters
    hyperparameters = torch.from_numpy(hyperparameters)

    CMA_CONTROLLER.mppi.noise_sigma = hyperparameters[0]*torch.eye(CMA_ENV.action_space.shape[0])
    CMA_CONTROLLER.mppi.noise_sigma_inv = torch.inverse(CMA_CONTROLLER.mppi.noise_sigma)
    CMA_CONTROLLER.mppi.noise_dist = MultivariateNormal(CMA_CONTROLLER.mppi.noise_mu, covariance_matrix=CMA_CONTROLLER.mppi.noise_sigma)
    CMA_CONTROLLER.mppi.lambda_ = hyperparameters[1]
    CMA_CONTROLLER.mppi.x_weight = hyperparameters[2]
    CMA_CONTROLLER.mppi.y_weight = hyperparameters[3]
    CMA_CONTROLLER.mppi.theta_weight = hyperparameters[4]

    #Simulate
    i, goal_distance, goal_reached = execute(CMA_ENV, CMA_CONTROLLER)
    
    #Retrieve cost
    cost = execution_cost(i, goal_distance, goal_reached)
    return cost 


def run_CMA(obsInit):
    CMA_ENV.obsInit = obsInit
    opts = cma.CMAOptions()
    opts.set("bounds", [[0, 0, 0, 0, 0], [None, 0.015, 10., 10., 10.]])
    res = cma.fmin(CMA_evaluate, [0.5, 0.01, 1, 1, 0.1], 1, opts)
    es = cma.CMAEvolutionStrategy([0.5, 0.01, 1, 1, 0.1], 1).optimize(CMA_evaluate)

def evalHP(hyperparameters, trials, obsInit):
    env = PandaPushingEnv(visualizer=None, render_non_push_motions=False,  include_obstacle=True,
                        camera_heigh=800, camera_width=800, render_every_n_steps=5, obsInit=obsInit)
    state_0 = env.reset()
    state = state_0
    controller = PushingController(env, pushing_multistep_residual_dynamics_model,
                                obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=30)
    
    controller.mppi.noise_sigma = hyperparameters[0]*torch.eye(env.action_space.shape[0])
    controller.mppi.noise_sigma_inv = torch.inverse(controller.mppi.noise_sigma)
    controller.mppi.noise_dist = MultivariateNormal(controller.mppi.noise_mu, covariance_matrix=controller.mppi.noise_sigma)
    controller.mppi.lambda_ = hyperparameters[1]
    controller.mppi.x_weight = hyperparameters[2]
    controller.mppi.y_weight = hyperparameters[3]
    controller.mppi.theta_weight = hyperparameters[4]
    pbar = tqdm(range(trials))
    success = 0
    for i in pbar:
        _, _, goal_reached = execute(env, controller)
        if goal_reached:
            success += 1
        pbar.set_description(f'Succeses: {success:.0f} | Total: {i+1:.0f}')
    successRate = success/trials
    print('Sucess Rate: ', successRate)

    return successRate

if __name__ == "__main__":
    colData = True
    trainGP = True
    OBS_INIT = 1
    TS = True
    if colData:
        collected_data = collect_data(OBS_INIT)
    train_x, train_y = load_data(OBS_INIT)
    if trainGP:
        train_gp(train_x, train_y, OBS_INIT)

    if TS:
        optimum_hp, optimum_cost = run_TS(train_x, train_y, OBS_INIT)
        print('Optimal HP: ', optimum_hp)
        print('Optimal Cost: ', optimum_cost)
    else:
        run_CMA(OBS_INIT)
