import pickle
import numpy as np
import sys
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import random
from tqdm import tqdm

import sys
sys.path.append('../')

# ==============================================================================

class FittedQIteration():
    """FittedQIteration is an implementation of the 
    Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).
    This class allows the use of a variety of regression algorithms, 
    provided by scikits-learn, to be used for representing the Q-value 
    function. Additionally, different basis functions can be applied 
    to the features before being passed to the regressors, including 
    trivial, fourier, tile coding, and radial basis functions. 
    NOTE: This algorithm assumes a discrete action space, though the
    state space does not necessarily have to be discrete (could be
    continuous, so long as the regressor can handle continuous features).
    """
    
    def __init__(self, 
                 gamma=0.90, 
                 iterations=40, 
                 action_space=[0, 1, 2, 3], 
                 state_space_dim=8, 
                 regressor=None, 
                 preset_params=None, 
                 ins=None,
                 perturb_rate=0.0, 
                 episode_length=200, 
                 cache=False,
                 use_running_history=False):
        """Inits the Fitted Q-Iteration planner with discount factor, 
        instantiated model learner, and additional parameters.
        
        Args:
            gamma: The discount factor for the domain
            regressor: Regression model class instance for the function 
                approximation. Needs to have fit() and predict() methods.
            **kwargs: Additional parameters for use in the class.
        """
        
        self.gamma = gamma
        self.iterations = iterations
        
        # Set up regressor
        self.regressor = regressor
        
        # TODO: Extend this idea to the gym env. action space
        self.action_space = action_space
        self.num_actions = len(action_space)
        
        # TODO: Extend this idea to gym env state space?
        self.state_space_dim = state_space_dim
        
        self.use_running_history = use_running_history
        
        # A record of episodes observed, to be used in the pseudo-batch setting
        self.running_history = []  
        
        # Decision: define a wrapper that takes in arguments of
        #      - Type of action space (discrete, continuous)
        #      - Type of model (linear, nonlinear)
        # and returns a function that takes state and action
        # and returns state X action
        
        # self.num_states = num_states
        # self.eps = 1.0
        # self.samples = None
        # self.preset_params = preset_params
        # self.ins = ins
        # self.episode_length = episode_length
        
        
    def encode_action(self, action):
        a = np.zeros(self.num_actions)
        a[action] = 1
        return a
    
    
    def get_episode(self, dataset, track=False):
        """Get a single episode from a dataset.
        
        Args:
            dataset: A DataSet object with a `get_next` method that
                returns the next episode in dictionary format (see
                the `get_next` and `get_episode` methods of HeparinDataset)
        """
        list_of_sars_tuples = []  # State, action, reward, next state
        
        epi_dict = dataset.get_next()
        if epi_dict is None:
            return None     # reached the end of the dataset.
        
        # TODO: Can we incorporate/use time information here?
            # But if you do pass in the time and mask information
            # then FQI should be able to use that effectively
            # (how to handle the training and the loss)
        epi_length = len(epi_dict['states'])
        for idx in range(0, epi_length):
            state = epi_dict['states'][idx]
            action = epi_dict['actions'][idx]
            reward = epi_dict['rewards'][idx]
            next_state = epi_dict['next_states'][idx]
            # info = epi_dict['infos'][idx]
            
            sars = np.hstack([state, action, reward, next_state])
            if self.use_running_history:
                self.running_history.append(sars)
            list_of_sars_tuples.append(sars)
        
        return list_of_sars_tuples
    
    
    def get_batch_episode(self, dataset, num_episodes=None):
        """Get a batch of episodes from a data set.
        
        Args:
            dataset: List of the dataset manager to import the data from
            num_episodes: The number of episodes to be returned in the batch. 
                If None, get all episodes.
        """
        ep_batch = []
        ep_idx = 0
        sars_tuple_list = self.get_episode(dataset, track=True)
        while sars_tuple_list is not None and (num_episodes is None or ep_idx < num_episodes):
            ep_batch.extend(sars_tuple_list)
            sars_tuple_list = self.get_episode(dataset, track=True)
            ep_idx += 1
        return np.vstack(ep_batch)  # TODO: Handle case where ep_batch is empty
    
    
    def get_batch_episodes_datasets(self, datasets, num_episodes=None, shuffle=True):
        """Get batches of episodes from multiple datasets 
        
        Can use a list of just a single dataset to get all episodes from that dataset
        """
        ep_batches = []
        for dataset in datasets:
            batch_of_episodes = self.get_batch_episode(dataset, num_episodes)
            ep_batches.append(batch_of_episodes)
        all_epis = np.concatenate(ep_batches)
        all_epis = np.array(all_epis, dtype=np.float32)  # TODO: This may be problematic. Double check that it plays nicely with openai gym action and state space representations
        
        # Note that there may be some states for which the next state is NaN.
        # We drop those rows in this case (we can't use them for FQI)
        all_epis = all_epis[~np.any(np.isnan(all_epis), axis=1), :]
        assert np.sum(np.isnan(all_epis)) == 0, 'NaN values found in episode batch'
        if shuffle:
            np.random.shuffle(all_epis)
        return all_epis
    
    
    def predictQs(self, states):
        """Get Q-value function evaluated for all possible actions given state.
        
        Args:
            states: The array of state features (for potentially multiple states)
            
        Returns:
            The double value for the value function at the given state
        """
        if len(state.shape) == 1:  # If a rank 0 array, 
            state = np.array(state).reshape(1, len(state))
        # Appends a column of the value `a` where `a` is the action for all
        # possible actions, and does this |A| times, returning |A| arrays,
        # each representing the state X action for different actions
        # TODO: This representation doesn't work 
        # Returns a matrix that is dim_action_space x N_states x (state_action_dim)
        Q = [
            self.regressor.predict(
                np.hstack([state, a * np.ones(len(state)).reshape(-1, 1)])  # TODO: This should call encode_action. Currently relies on the assumption. This SHOULD be of dimension |S x A|, which may not necessarily be equal to |S| + 1
            ) for a in self.action_space
        ]  # This is a list of arrays of length action_dim, each of which is N_states x 1
        return Q  # This is action_dim x N_states
    
    
    def predictMaxQ(self, state):
        """Get Q-value function value greedy action at given state (ie V(state)).
        
        Args:
            state: The array of state features
            
        Returns:
            The double value for the value function at the given state
        """
        Q = self.predictQs(state)  # dimensions: action_dim x num_states
        return np.amax(Q, axis=0)  # dimensions: num_states x 1
    
    
    def policy(self, state, eps=0.0):
        """Get the action under the current plan policy for the given state.
        
        Args:
            state: The array of state features
        Returns:
            The current greedy action under the planned policy for the given state. 
            If no plan has been formed, return a random action.
        """
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.num_actions)
        else:
            Q = self.predictQs(state)
            return np.argmax(Q, axis=0)
        
        
    def train(self, env=None, datasets=None, save_path=None, num_episodes=None):
        """Learn optimal policy from data using Fitted Q-Iteration
        """
        if env is None and datasets is None:
            print("ERROR: either of env or dataset should be specified.")
            return None
        
        # Get all episodes for which the action taken was in our action space
        # Define a separate function to get a set of all the episodes from a dataset
        samples = self.get_batch_episodes_datasets(datasets, num_episodes=num_episodes)
        # samples[np.isin(samples[:, self.state_space_dim], self.action_space), :]
        # print("All samples : {}".format(samples.shape))
        
        Q = np.zeros(samples.shape[0])
        # TODO: Create a template for neural network fitting that returns lambda function with relevant 
        # fitting arguments initialized appropriately/according to user
        self.regressor.fit(samples[:, :self.state_space_dim + 1], Q)  # Think about creating a wrapper for neural nets?
        Q_history = [Q]
        
        for _ in tqdm(range(self.iterations)):
            Qprime = self.predictMaxQ(samples[:, -self.state_space_dim:])
            Q = samples[:, self.state_space_dim + 1] + Qprime * self.gamma
            self.regressor.fit(samples[:, :self.state_space_dim + 1], Q)
            Q_history.append(Q)
        
        if save_path is not None:
            print("saving Q function at {}".format(save_path))
            joblib.dump((self.num_actions, self.regressor), save_path)
            
        return np.array(Q_history)