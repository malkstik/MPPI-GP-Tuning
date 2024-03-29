import gpytorch
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from learning_state_dynamics import TARGET_POSE_OBSTACLES, BOX_SIZE, ResidualDynamicsModel, PushingController, obstacle_avoidance_pushing_cost_function
import numpy as np
from panda_pushing_env import *
from tqdm import tqdm
import cma 

#GAME PLAN

def execute(env, controller, num_steps_max = 30):
    state = env.reset()
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

def execution_cost(i, goal_distance, goal_reached):
    # A far distance is 0.4, A pass can get something like .06
    # cost = i + 10*goal_distance 
    # if not goal_reached:
    #     cost += 30
    cost = goal_distance**2
    return cost

def collect_data_GP(env, controller, dataset_size = 300):
    # --- Your code here
    collected_data = []
    pbar = tqdm(range(dataset_size))

    for i in pbar:
        pbar.set_description(f'Iteration: {i:.0f}')
        state_0 = env.reset()
        data = {}
        # data['hyperparameters'] = torch.zeros(5, dtype = torch.float32) #noise_sigma, lambda_value, x, y, theta
        data['cost'] = 0
        # Randomly Sample Hyperparameter values
        data['hyperparameters'] = np.random.uniform(0, 10)
        # data['hyperparameters'][0] = np.random.uniform(0, 10)
        # data['hyperparameters'][1] = np.random.uniform(0, 0.015)
        # data['hyperparameters'][2] = np.random.uniform(0, 10)
        # data['hyperparameters'][3] = np.random.uniform(0, 10)
        # data['hyperparameters'][4] = np.random.uniform(0, 10)

        # a = np.array([0, 0, 0, 0])
        # b = np.array([10, 10, 10, 10])
        # data['hyperparameters'] = np.random.uniform(a, b)

        #Should we also consider changing horizon?
        # Simulate using these hyperparameters
        controller.mppi.noise_sigma = data['hyperparameters']*torch.eye(env.action_space.shape[0])
        controller.mppi.noise_sigma_inv = torch.inverse(controller.mppi.noise_sigma)        
        controller.mppi.noise_dist = MultivariateNormal(controller.mppi.noise_mu, covariance_matrix= controller.mppi.noise_sigma)
        # controller.mppi.lambda_ = data['hyperparameters'][1]
        # controller.mppi.x_weight = data['hyperparameters'][2]
        # controller.mppi.y_weight = data['hyperparameters'][3]
        # controller.mppi.theta_weight = data['hyperparameters'][4]
        # Add cost to data
        cost = 0
        # for i in range(5):
        steps, goal_distance, goal_reached = execute(env, controller)
        cost += execution_cost(steps, goal_distance, goal_reached)
        data['cost'] = cost
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
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_nums_dims = 1)
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
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.forward(x))
        pred_mu = pred.mean
        print(x.shape)
        print(pred.mean.shape)
        # upper, _ = pred.confidence_region()
        # pred_sigma = (upper - pred)/2
        pred_sigma = pred.stddev#torch.diag_embed(pred.stddev ** 2)

        return pred_mu, pred_sigma

def train_gp_hyperparams(model, likelihood, hyperparameters, cost, lr = 0.1, mute = False):
    
    """
        Function which optimizes the GP Kernel & likelihood hyperparameters
    Args:
        model: gpytorch.model.ExactGP model
        likelihood: gpytorch likelihood
        hyperparameters: (N, 4) torch.tensor of hyperparameters
        cost: (N,) torch.tensor of cost due to hyperparameters
        lr: Learning rate

    """
    # --- Your code here
    training_iter = 2000
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
        if not mute:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    # ---

class ThompsonSamplingGP:
    
    def __init__(self, state_dict, likelihood, constraints, dynamics_model, train_x, train_y,
                prior = None, n_random_draws = 5, interval_resolution=500, device = 'cpu', obsInit = 0):
                
        # number of random samples before starting the optimization
        self.n_random_draws = n_random_draws 
        
        # the bounds tell us the interval of x we can work
        self.constraints = constraints
        
        # interval resolution is defined as how many points we will use to 
        # represent the posterior sample
        # we also define the x grid
        self.interval_resolution = interval_resolution

        # self.vals = torch.tensor([.5, .01, 1, 1, 0.1]).to(torch.device(device))
        # self.X_grid = self.vals.clone().repeat((interval_resolution,1)).to(torch.device(device))
        self.X_grid = torch.linspace(constraints[0,0], constraints[0,1], self.interval_resolution).type(torch.float64)
        # self.hp_focus = 0

        # for i in range(5):
        #     # self.X_grid[:,i] = torch.linspace(self.constraints[i, 0], self.constraints[i, 1], self.interval_resolution)
        #     self.X_grid[:,i] *= default[i]

        # also initializing our design matrix and target variable
        if prior is not None:
            self.X = prior[0].type(torch.float64)
            self.y = prior[1].type(torch.float64)
            self.first = False
            print('prior is not none')
        else:
            self.first = True
            self.X = torch.tensor([]).to(torch.device(device)).type(torch.float64); self.y = torch.tensor([]).to(torch.device(device)).type(torch.float64)
            print('prior is none')
        # parameters for fitting a GP
        self.state_dict = state_dict
        self.likelihood = likelihood
        self.likelihood.eval()

        # create environment and controller for evaluating cost
        self.env = PandaPushingEnv(visualizer=None, render_non_push_motions=False,  include_obstacle=True,
                                    camera_heigh=800, camera_width=800, render_every_n_steps=20, obsInit = obsInit)
        self.env.reset()
        self.controller = PushingController(self.env, dynamics_model,
                        obstacle_avoidance_pushing_cost_function, num_samples=100, horizon=30)

        # device to run on
        self.device = device

        # store training data for retraining
        self.train_x = train_x
        self.train_y = train_y


    def fit(self, X, y):
        # print(X.shape, y.shape)
        if (self.n+1)%3 == 0:
            retrain_HP = torch.hstack((self.train_x, X)).type(torch.float64)
            retrain_cost = torch.hstack((self.train_y, y)).type(torch.float64)
            gp_model = RBF_GP(retrain_HP, retrain_cost, self.likelihood)
            self.refineGP(gp_model, self.likelihood, retrain_HP, retrain_cost)
        else:
            gp_model = RBF_GP(X, y, self.likelihood)
            gp_model.load_state_dict(self.state_dict)
            gp_model.eval()
        return gp_model

    def evaluate(self, sample):
        #Change controller hyperparameters
        self.controller.mppi.noise_sigma = sample*torch.eye(self.env.action_space.shape[0])
        self.controller.mppi.noise_sigma_inv = torch.inverse(self.controller.mppi.noise_sigma)
        self.controller.mppi.noise_dist = MultivariateNormal(self.controller.mppi.noise_mu, covariance_matrix=self.controller.mppi.noise_sigma)
        # self.controller.mppi.lambda_ = sample[1]
        # self.controller.mppi.x_weight = sample[2]
        # self.controller.mppi.y_weight = sample[3]
        # self.controller.mppi.theta_weight = sample[4]

        #Simulate
        i, goal_distance, goal_reached = execute(self.env, self.controller)
        cost = execution_cost(i, goal_distance, goal_reached)
        #Averaging performs well but is too computationally expensive
        # cost = 0
        # for j in range(5):
        #     i, goal_distance, goal_reached = execute(self.env, self.controller)
        #     #Retrieve cost
        #     cost += execution_cost(i, goal_distance, goal_reached)
        # cost /= 5
        return cost 

    # process of choosing next point
    def choose_next_sample(self):
        # if we do not have enough samples, sample randomly from bounds
        if self.X.shape[0] < self.n_random_draws:
            # next_sample = np.random.uniform(self.constraints[:,0], self.constraints[:,1])
            next_sample = np.random.uniform(self.constraints[0,0], self.constraints[0,1])

            # next_sample = torch.from_numpy(next_sample).to(torch.device(self.device))
            next_sample = torch.tensor([next_sample]).to(torch.device(self.device))

        # if we do, we fit the GP and choose the next point based on the posterior draw minimum
        else:
            # 1. Fit the GP and draw one sample (a function) from the posterior
            # print(self.X.shape)
            # print(self.y.shape)
            # print(self.X.dtype)
            # print(self.y.dtype)
            # print(self.X_grid.dtype)
            gp_model = self.fit(self.X, self.y)
            #self.make_grid() #Cycles through hyperparameter of interest, reduce problem to 1D at any one given time
            # posterior_mean, posterior_std = gp_model.predict(self.X_grid)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(gp_model(self.X_grid))
                posterior_mean = observed_pred.mean
                posterior_mean.shape
                posterior_std =  observed_pred.stddev
            posterior_sample = posterior_mean + posterior_std*torch.randn_like(posterior_mean)

            posterior_sample = posterior_sample.to(torch.device(self.device))

            # 2. Choose next point as the optimum of the sample
            which_min = torch.argmin(posterior_sample)
            # next_sample = self.X_grid[which_min, :]
            next_sample = self.X_grid[which_min]
            self.vals = next_sample.clone()
        
        # let us observe the objective and append this new data to our X and y
        next_observation = torch.tensor(self.evaluate(next_sample))
        if self.first:
            self.X = next_sample.type(torch.float64)
            self.y = next_observation.type(torch.float64)
            self.first = False
            # print('first')
        else:
            self.X = torch.hstack((self.X, next_sample))
            self.y = torch.hstack((self.y, next_observation))
            # print('not first')
      
    def getOptimalParameters(self, iter = 100):
        pbar = tqdm(range(iter))
        for self.n in pbar:
            self.choose_next_sample()
        bestIdx = torch.argmin(self.y)
        return self.X[bestIdx], self.y[bestIdx]
    
    def refineGP(self, model, likelihood, retrain_HP, retrain_cost):
        train_gp_hyperparams(model, likelihood, retrain_HP, retrain_cost, mute=True)
        self.state_dict = model.state_dict()

    def make_grid(self):
        thisgrid = self.vals.clone()
        self.X_grid = self.vals.clone().repeat((self.interval_resolution,1)).to(torch.device(self.device))
        hp = self.hp_focus%5
        if hp in [0, 2, 3, 4]:
            self.X_grid[:, hp] = torch.linspace(self.constraints[hp,0], self.constraints[hp,1], self.interval_resolution)
            if self.n%5 == 0:
                self.hp_focus += 1
        # Skipping lambda_val since it doesnt do anything
        else:
            self.X_grid[:, 2] = torch.linspace(self.constraints[2,0], self.constraints[2,1], self.interval_resolution)
            if self.n%5 == 0:
                self.hp_focus += 2
