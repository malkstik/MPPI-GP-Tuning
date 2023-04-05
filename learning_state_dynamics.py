import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, BOX_SIZE
TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
#OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]

def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
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
    state_size = env.observation_space.sample().shape[0]
    actions_size = env.action_space.sample().shape[0]

    for i in range(num_trajectories):
      data = {}
      data['states'] = np.zeros((trajectory_length + 1, state_size), dtype = np.float32)
      data['actions'] = np.zeros((trajectory_length, actions_size), dtype = np.float32)
      data['states'][0,:] = env.reset()
      for j in range(trajectory_length):
        action_j = env.action_space.sample()
        state, _, _, _ = env.step(action_j)
        data['states'][j+1,:] = state
        data['actions'][j,:] = action_j
      collected_data.append(data)
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


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    total_Dataset = MultiStepDynamicsDataset(collected_data, num_steps)
    train_set_size = int(0.8*len(total_Dataset))
    val_set_size = len(total_Dataset) - train_set_size

    train_set, val_set = random_split(total_Dataset, [train_set_size, val_set_size])

    train_loader = DataLoader(train_set, batch_size = batch_size)
    val_loader = DataLoader(val_set, batch_size = batch_size)

    # ---
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here
        #100 Trajectories, 10 Trajectory length
        #item can go between 0 and 999
        #consider stacking trajectories: T1_0 ... T1_9 T2_0 ...
        # Then first 10 capture first trajectory, second 10 capture second trajectory
        # int(item/10) should indicate which trajectory we're in
        # item%10 shouldp provide which step of the trajectory we're in
        trajectory = int(item/self.trajectory_length)
        step = item%self.trajectory_length

        sample['state'] = self.data[trajectory]['states'][step, :]
        sample['action'] = self.data[trajectory]['actions'][step, :]
        try:
          sample['next_state'] = self.data[trajectory]['states'][step+1, :]
        except:
          print('There was no next state')
          sample['next_state'] = None

        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here
        trajectory = int(item/self.trajectory_length)
        step = item%self.trajectory_length

        sample['state'] = self.data[trajectory]['states'][step, :]
        sample['action'] = self.data[trajectory]['actions'][step:(step+self.num_steps), :]
        try:
          sample['next_state'] = self.data[trajectory]['states'][step+1:(step+1+self.num_steps), :]
        except:
          print('There was no next state')
          sample['next_state'] = None
        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):

        se2_pose_loss = None
        # --- Your code here
        rg = ((self.l**2 + self.w**2)/12)**0.5
        se2_pose_loss = (F.mse_loss(pose_pred[:,0], pose_target[:,0])+
                        F.mse_loss(pose_pred[:,1], pose_target[:,1])+
                        rg*F.mse_loss(pose_pred[:,2], pose_target[:,2]))
        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        pred_state = model.forward(state,action)
        single_step_loss = self.loss.forward(pred_state, target_state)
        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        multi_step_loss = 0
        num_steps = actions.shape[1]
        pred_state = state
        for step in range(num_steps):
          action = actions[:,step,:]
          pred_state = model.forward(pred_state, action)
          multi_step_loss += (self.discount**step)*self.loss.forward(pred_state, target_states[:,step,:])
        # ---
        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.lin1 = nn.Linear(self.state_dim + self.action_dim,100)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(100,100)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(100,self.state_dim)
        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        state_action = torch.cat((state,action), -1)
        next_state = self.lin1(state_action)
        next_state = self.act1(next_state)
        next_state = self.lin2(next_state)
        next_state = self.act2(next_state)
        next_state = self.lin3(next_state)
        # ---
        return next_state


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.lin1 = nn.Linear(self.state_dim + self.action_dim,100)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(100,100)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(100,self.state_dim)
        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        state_action = torch.cat((state,action), -1)
        delta = self.lin1(state_action)
        delta = self.act1(delta)
        delta = self.lin2(delta)
        delta = self.act2(delta)
        delta = self.lin3(delta)
        next_state = delta + state
        # ---
        return next_state


def free_pushing_cost_function(state, action, Q):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    B = state.shape[0]
    cost = torch.zeros(B)
    Q = torch.from_numpy(Q)
    state_diff = torch.unsqueeze((state-target_pose), -1)
    cost = torch.sum(torch.einsum('bij,jk,bik->bi', state_diff, Q, state_diff), dim=1)
    #Rule out impossible actions
    # if not env.check_action_valid(action.detach().cpu().numpy()):
    #     cost = 200
    # ---
    return cost

def collision_detection(state, obs_centre):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = torch.from_numpy(obs_centre[:2])  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_halfdim = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    w_obs, l_obs = OBSTACLE_HALFDIMS_TENSOR
    B = state.shape[0]
    
    in_collision = torch.ones(B, device=state.device) 

    # Check those close enough to fail 1 with Separating Axis Theorem
    to_corner = lambda w, l, theta: torch.hstack([w*torch.cos(theta)-l*torch.sin(theta),
                                                  w*torch.sin(theta)+l*torch.cos(theta)])

    zero_tensor = torch.zeros(1, device=state.device)

    obs_verts = torch.hstack((obstacle_centre + to_corner(w_obs, l_obs, zero_tensor),
                              obstacle_centre + to_corner(w_obs, -l_obs, zero_tensor),
                              obstacle_centre + to_corner(-w_obs, -l_obs, zero_tensor),
                              obstacle_centre + to_corner(-w_obs, l_obs, zero_tensor)
                              )).reshape(-1, 4, 2)
    

    state_verts = torch.hstack((state[:,:2] + to_corner(box_halfdim, box_halfdim, state[:,2:]),
                                state[:,:2] + to_corner(box_halfdim, -box_halfdim, state[:,2:]),
                                state[:,:2] + to_corner(-box_halfdim, -box_halfdim, state[:,2:]),
                                state[:,:2] + to_corner(-box_halfdim, box_halfdim, state[:,2:])
                                )).reshape(-1, 4, 2)
    
    # obstacle collide with box
    si_Minus = state_verts[:,-1] - state_verts[:,0] # dims are (B, 2)
    for i in range(state_verts.shape[1]):
        si_Plus = torch.roll(state_verts, -i, 1)[:,1] - torch.roll(state_verts, -i, 1)[:,0] # the proposed seperating axis
        ni = torch.cat([-si_Plus[:,1:], si_Plus[:,:1]],dim=-1) # the normal to the proposed separating axis
        sgni = torch.sign(torch.sum(si_Minus * ni, dim=1)) # side that rect A is on    

        prop_axis = torch.ones_like(in_collision) # becomes 0 if axis ruled out (if any not ruled out then no collision)
        for j in range(obs_verts.shape[1]):
            sij = torch.roll(obs_verts, -j, 1)[:,0] - torch.roll(state_verts, -i, 1)[:,0]
            sgnj = torch.sign(torch.sum(sij * ni, dim=1))

            prop_axis[sgni * sgnj > 0] = 0 # if they have same sign then rule out axis candidate

        in_collision[prop_axis == 1] = 0 # Found seperating axis: no j vert ruled out i's edge
        si_Minus = -si_Plus
    
    # box collide with obstacle
    si_Minus = obs_verts[:,-1] - obs_verts[:,0] # dims are (B, 2)
    for i in range(obs_verts.shape[1]):
        si_Plus = torch.roll(obs_verts, -i, 1)[:,1] - torch.roll(obs_verts, -i, 1)[:,0] # the proposed seperating axis
        ni = torch.cat([-si_Plus[:,1:], si_Plus[:,:1]],dim=-1) # the normal to the proposed separating axis
        sgni = torch.sign(torch.sum(si_Minus * ni, dim=1)) # side that rect A is on        

        temp_axis = torch.ones_like(in_collision)
        for j in range(state_verts.shape[1]):
            sij = torch.roll(state_verts, -j, 1)[:,0] - torch.roll(obs_verts, -i, 1)[:,0]
            sgnj = torch.sign(torch.sum(sij * ni, dim=1))

            temp_axis[sgni * sgnj > 0] = 0
                
        in_collision[temp_axis == 1] = 0 # Found seperating axis: no j vert ruled out i's edge
        si_Minus = -si_Plus

    # ---
    return in_collision

def obstacle_avoidance_pushing_cost_function(state, action, OBSTACLE_CENTRE, Q):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    B = action.shape[0]
    cost = free_pushing_cost_function(state, action, Q)
    
    #Check for collisions and penalize
    in_collision = torch.zeros((B), dtype= bool)
    for j in range(5):
        in_collision = torch.logical_or(in_collision, collision_detection(state, OBSTACLE_CENTRE[j]))
    in_collision = torch.where(in_collision, 1, 0).type(torch.float)
    # if torch.min(in_collision) >0:
    #     print("Couldn't find one without colliding")
    cost += 100*in_collision
    # ---
    return cost

class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        print(u_min)
        print(u_max)
        noise_sigma = 0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function, 
                         state_dim, 
                         noise_sigma,
                         env.OBSTACLE_CENTRE,
                         num_samples=num_samples,
                         horizon=horizon,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max
        )

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        next_state = self.model(state,action)
        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state)
        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.cpu().detach().numpy()
        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here
def train_step(model, loss_model, train_loader, optimizer) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0.0
    model.train()
    for batch_idx, data in enumerate(train_loader):     
      optimizer.zero_grad()
      loss = loss_model.forward(model, data['state'], data['action'], data['next_state'])        
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    return train_loss/len(train_loader)

def val_step(model, loss_model, val_loader) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0.0
    model.eval()
    for batch_idx, data in enumerate(val_loader):
      loss = loss_model.forward(model, data['state'], data['action'], data['next_state']) 
      val_loss += loss.item()
    return val_loss/len(val_loader)

def train_model(model, loss_model, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = train_step(model, loss_model, train_dataloader, optimizer)
        val_loss_i = val_step(model, loss_model, val_dataloader)
        pbar.set_description(f'Train Loss: {train_loss_i:.8f} | Validation Loss: {val_loss_i:.8f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses  
# ---
# ============================================================
