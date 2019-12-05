import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
from sklearn.impute import IterativeImputer
from tqdm import tqdm

class HeparinDataset:
    """A dataset that maintains a copy of 
    """
    
    def __init__(self, 
                 df,
                 state_names,
                 next_state_names,
                 action_space=np.array([0, 1, 2, 3]),
                 imputer=None,
                 action_name='action',
                 reward_name='reward',
                 ep_id_name='episode',
                 num_state_history_to_include_in_state=1,
                 num_action_history_to_include_in_state=1,
                 discount_factor=0.99, 
                 p_train=0.8,
                 seed=42,
                 shuffle=True,
                 from_scratch=False):
        
        self.data = df
        self.state_names = state_names
        self.action_name = action_name
        self.action_space = action_space
        self.reward_name = reward_name
        self.next_state_names = next_state_names
        self.ep_id_name = ep_id_name

        if not imputer:  # TODO: Handle the case where no imputation is needed
            imputer = IterativeImputer(missing_values=np.nan,
                                       random_state=seed,
                                       max_iter=30,
                                       tol=1e-3,
                                       n_nearest_features=None)
            imputer.fit(df[state_names])
        df[state_names] = imputer.transform(df[state_names])
        self.imputer = imputer
        
        self.all_ep_ids = np.unique(df[ep_id_name])
        self.current_idx = 0
        self.reset(shuffle=shuffle)
#         df = df.loc[
#             np.isin(
#                 np.array(df[self.action_name], dtype=np.int), 
#                 np.array(self.action_space)
#             ), :]
        self.data = df
    
    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.all_ep_ids)
        self.current_idx = 0
    
    
    def get_next(self):
        """
        Return the next episode.
        
        This function is called by fittedQiter.py, for example
        """
        if self.current_idx >= len(self.all_ep_ids):
            return None
        
        epi = self.get_episode(self.all_ep_ids[self.current_idx])
        self.current_idx += 1
        return epi
    
    
    def get_all_ep_ids(self):
        return self.all_ep_ids
    
    
    def get_episode(self, ep_id=None):
        """
        Return an episode of the given ep_id. If ep_id is not given (or None),
        return a randomly selected episode.
        """
        
        states = []
        actions = []
        rewards = []
        next_states = []
        infos = []
        reward_sum = np.array([0.0])
        
        if ep_id is None:
            ep_id = np.random.choice(self.all_ep_ids)
        
        assert ep_id in self.all_ep_ids
        
        self.data.reset_index()
        episode = self.data.loc[self.data[self.ep_id_name] == ep_id]
        
        for indx in episode.index:
            curr_state = episode.loc[indx, self.state_names]
            action = int(episode.loc[indx, self.action_name])
            reward = int(episode.loc[indx, self.reward_name])
            if indx + 1 in episode.index:
                next_state = episode.loc[indx + 1, self.next_state_names]
            else:
                next_state = np.repeat(np.nan, len(self.next_state_names))
            states.append(np.array(curr_state))
            actions.append(np.array([action]))
            rewards.append(np.array([reward]))
            next_states.append(np.array(next_state))
            reward_sum += reward
        
        infos = np.array([''])
        episode_returns = reward_sum  # Consider making this G_t instead (more useful)
        
        episode_starts = np.array([False for _ in range(0, len(states))])
        episode_ends = np.array([False for _ in range(0, len(states))])
        
        if len(episode_starts) > 0:
            episode_starts[0] = True
        if len(episode_ends) > 0:
            episode_ends[-1] = True
        
        episode_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'infos': infos,
            'episode_starts': episode_starts,
            'episode_ends': episode_ends,
            'episode_returns': reward_sum
        }
        return episode_dict
    
    
    def get_episodes(self, ep_id_list=None, save_path=None):
        """
        Generate numpy dict (or .npz file if save_path is not None) from the trace files,
        each containing rows of (state, action, reward, next_state).
        
        Parameters:
            pid_list (List[int]): List of patient_id's to get the episodes. 
                If null, get all available ones.
            save_path (str): file path to save the results in .npz format. 
                If None, results are not saved in file.
        """
        
        # If no ep_id_list is specified, just load everything
        
        if ep_id_list is None:
            ep_id_list = self.all_ep_ids
            
        all_episodes = None
        for ep_id in tqdm(ep_id_list):
            episode = self.get_episode(ep_id)
            if all_episodes is None:
                all_episodes = episode
            else:
                # print('all_episodes = {}'.format(all_episodes))
                self._merge_episode_dict(all_episodes, episode)
                
        if save_path is not None:
            np.savez(save_path, **all_episodes)
            
        return all_episodes
    
    
    def _merge_episode_dict(self, all_episodes, episode):
        for key in all_episodes:
            all_episodes[key] = np.append(all_episodes[key], episode[key], axis=0)
    
    
    def write_to_file(self, df, path):
        df.to_csv(path, index=False)