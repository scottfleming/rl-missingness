import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from tqdm import tqdm
import os
from datetime import datetime
import random
import re

REWARD_THERAPEUTIC_XA = 1
REWARD_THERAPEUTIC_PTT = 1

def hours_between(t1_str, t2_str, str_format='%Y-%m-%d %H:%M:%S'):
    t1 = datetime.strptime(t1_str, str_format)
    t2 = datetime.strptime(t2_str, str_format)
    time_diff_seconds = abs((t2 - t1).total_seconds())
    time_diff_hrs = time_diff_seconds / 60. / 60.
    return time_diff_hrs
    
    
def one_hot_encode(df, columns_to_one_hot):
    import category_encoders as ce  # Nonstandard package, only if needed
    
    one_hot_encoder = ce.OneHotEncoder(cols=columns_to_one_hot, 
                                       use_cat_names=True)
    one_hot_encoder.fit(df)
    df = one_hot_encoder.transform(df)
    
    # The category_encoders package creates extra empty columns
    # for some reason, so we just delete them in the end
    
    for col in df.columns:
        if re.search('_-1', col):
            df.drop(col, axis=1, inplace=True)
    return df

    
def binarize(df, columns_to_binarize):
    for col in columns_to_binarize:
        if col in df.columns:
            df[col] = 1.0 * df[col]
    return df


class HeparinDataProcessor:
    """A dataset that maintains a copy of 
    """
    
    def __init__(self, 
                 filepath, 
                 columns_to_impute,
                 columns_to_drop,
                 columns_to_one_hot,
                 columns_to_binarize,
                 action_name='action',
                 reward_name='reward',
                 num_state_history_to_include_in_state=1,
                 num_action_history_to_include_in_state=2,
                 discount_factor=0.99, 
                 p_train=0.8,
                 seed=12345,
                 shuffle=True,
                 from_scratch=False):
        
        self.path = filepath
        self.discount_factor = discount_factor
        self.data = pd.read_csv(filepath)
        
        self.action_name = action_name
        self.reward_name = reward_name
        
        # Replace all missing heparin dose values with 0.
        # (We assume that if heparin administration wasn't recorded, 
        # then it wasn't given)
        
        self.data['HEPARIN_DOSE'] = self.data['HEPARIN_DOSE'].fillna(0)
        
        # Drop all patients who were receiving LMWH (i.e. 
        # drop where LMWH_LEVEL is NOT NaN) because we're only interested
        # in patients who are exclusively on unfractionated heparin
        
        self.data = self.data.loc[self.data['LMWH_LEVEL'].isna(), :]
        self.data = self.data.drop('LMWH_LEVEL', axis=1)
        
        # Drop all duplicate entries (e.g. recorded at the exact same time), 
        # keeping just the last entry. Aberration of our specific dataset.
        
        self.data = self.data.drop_duplicates(subset=['ANON_ID', 'Date'], 
                                              keep='last')
        self.data = self.data.reset_index()
        self.data = self.data.drop(columns=['index'])
        self.num_state_history_to_include_in_state = num_state_history_to_include_in_state
        self.num_action_history_to_include_in_state = num_action_history_to_include_in_state
        
        self.augmented_data = self.data.copy()
        if (os.path.isfile('data/augmented_data.csv') and not from_scratch):
            self.augmented = pd.read_csv('data/augmented_data.csv')
        else:
            print("Calculating Rewards...")
            self.augmented_data = self.add_reward_columns(self.augmented_data)
            print("Calculating Actions...")
            self.augmented_data = self.augment_with_missingness_info(
                self.augmented_data,
                columns_to_impute
            )
            
            self.augmented_data = one_hot_encode(self.augmented_data, 
                                                 columns_to_one_hot)
            self.augmented_data = binarize(self.augmented_data, 
                                           columns_to_binarize)
            self.augmented_data = self.augmented_data.loc[
                ~pd.isnull(self.augmented_data[self.action_name])
            ]
            self.write_to_file(self.augmented_data, 'data/augmented_data.csv')
        
        
    def write_to_file(self, df, path):
        df.to_csv(path, index=False)
        
    
    def generate_k_patient_id_folds(self, unique_patient_ids, k, seed=42):
        """Randomly generate K disjoint train/test folds of (unique) patient IDs
        
        Args:
            unique_patient_ids: A list of unique patient IDs that should be
                split into five different folds
        
        Returns:
            Dictionary wherein the key is the fold name e.g. "fold1", "fold2", etc.
                and the value is another dictionary whose keys "train" and "test"
                correspond to patient IDs within that particular fold.
        """
        assert len(np.unique(unique_patient_ids)) == len(unique_patient_ids)
        np.random.seed(seed)
        kf = KFold(n_splits=k, shuffle=True)
        patient_ids = {}
        for i, (train_indxs, test_indxs) in enumerate(kf.split(unique_patient_ids)):
            fold_name = 'fold' + str(i + 1)
            patient_ids[fold_name] = {}
            patient_ids[fold_name]['train'] = unique_patient_ids[train_indxs]
            patient_ids[fold_name]['test'] = unique_patient_ids[test_indxs]
        return patient_ids
    
    
    def add_reward_columns(self, df,
                           reward_therapeutic_Xa=REWARD_THERAPEUTIC_XA,
                           reward_therapeutic_PTT=REWARD_THERAPEUTIC_PTT):
        """Add columns with calculated rewards
        
        Args:
            df: pandas dataframe with (num_observations X num_features) dimensions
                with a column 'PTT' indicating observed PTT for each observation and 
                a column 'HAL' 
            reward_therapeutic_Xa: integer reward associated with being in the 
                therapeutic range for anti-Xa
            reward_therapeutic_PTT: integer reward associated with being in the
                therapeutic range for aPTT
                
        Returns:
            df with columns 'PTT_score', 'Xa_score', and 'PTT_Xa_score' indicating
                the scores associated 
        """
        
        df['PTT_good'] = (df['PTT'] >= 40) & (df['PTT'] <= 80)
        df['Xa_good'] = (df['HAL'] >= 0.3) & (df['HAL'] <= 0.7)
        df['PTT_score'] = df['PTT_good'] * reward_therapeutic_PTT
        df['Xa_score'] = df['Xa_good'] * reward_therapeutic_Xa
        
        # Note that for this experiment we shift everything to be negative
        df['Xa_score'] = df['Xa_score'] - np.max(df['Xa_score'])
        df['PTT_score'] = df['PTT_score'] - np.max(df['PTT_score'])
        df['PTT_Xa_score'] = df['PTT_score'] + df['Xa_score']
        
        df['reward'] = df['Xa_score'] + df['PTT_score'] - 1
        
        return df
    
    
    def get_action(self, next_24hr_heparin, curr_24hr_heparin):
        """Calculate coded action in accordance with current/next 24hr heparin
        
        Args:
            next_24hr_heparin: Cumulative heparin given over next 24 hours
            curr_24hr_heparin: Cumulative heparin given over last 24 hours
            
        Returns:
            integer corresponding to
                0 if heparin dose was increased
                1 if heparin dose was zero and is still zero
                2 if heparin dose was nonzero and remains the same
                3 if heparin dose was increased
                
        Raises:
            ValueError if for some reason the action isn't recognized
        """
        if next_24hr_heparin > curr_24hr_heparin:
            action = 3  # Action taken was to increase heparin dosage
        elif next_24hr_heparin == curr_24hr_heparin and curr_24hr_heparin > 0:
            action = 2  # Action taken was to maintain positive heparin dosage
        elif next_24hr_heparin == curr_24hr_heparin and curr_24hr_heparin == 0:
            action = 1  # Action taken was to maintain zero heparin dosage
        elif next_24hr_heparin < curr_24hr_heparin:
            action = 0  # Action taken was to decrease heparin dosage
        else:
            raise ValueError('Uknown action type: ',
                             'dose = {}'.format(curr_24hr_heparin),
                             'dose_next = {}'.format(next_24hr_heparin))
        return action
    
    
    def augment_with_missingness_info(self, df, 
                                      columns_to_impute,
                                      min_obs_diff=0., 
                                      max_obs_diff=26.):
        """Add columns indicating step index and which action for each obs.
        
        Args: 
            df: pandas dataframe with (num_observations X num_features) dimensions
                and actions pre-defined for each obs.
            min_obs_diff (optional): Any observations that are less than min_obs_diff
                hours apart will be considered as belonging to separate trajectories
            max_obs_diff (optional): Any observations that are more than max_obs_diff
                hours apart will be considered as belonging to separate trajectories
                
        Returns:
            The original dataframe with additional columns:
                'action' corresponding to a numerically-coded action corresponding to the 
                    output from the get_action method. Note that the action is only well-defined 
                    if we can see the difference between the heparin dose at the next time step 
                    (because we only have access to how much heparin was received in the 
                    previous 24-hour period)
                'TRAJ_ID' corresponding to a unique trajectory ID for each patient trajectory
                'traj_num' indicating the serial order of the patient trajectories e.g.
                    a value of 1 in the column indicates it's the first observed trajectory
                    for that patient, a value of 2 indicates it's the second, and so on
                'obs_num' indicating the serial order of the patient observation in the trajectory
                
        """
        
        traj_id = 1  # A unique ID for the patient's trajectory
        traj_indx = 1  # 1 if 1st trajectory for patient, 2 if 2nd, etc.
        obs_indx = 1  # 1 if 1st obs. in patient traj., 2 if 2nd, etc.
        time = 0.  # Observation time normalized so that start of traj is 0
        
        df.loc[:, 'time_since_last_obs'] = 999
        curr_val_is_missing = {col:True for col in columns_to_impute}
        aug_cols = [col + '_t_since_last' for col in columns_to_impute]
        
        for col in columns_to_impute:
            df[col + '_t_since_last'] = 999
        
        for index in tqdm(df.index, total=df.shape[0]):
            if index != len(df) - 1:
                df.loc[index, 'TRAJ_ID'] = traj_id  
                df.loc[index, 'traj_num'] = traj_indx  
                df.loc[index, 'obs_num'] = obs_indx  
                
                if df.loc[index]['ANON_ID'] == df.loc[index + 1]['ANON_ID']:
                
                    curr_aPTT = df.loc[index]['PTT']
                    next_aPTT = df.loc[index + 1]['PTT']
                    
                    curr_HAL = df.loc[index]['HAL']
                    next_HAL = df.loc[index + 1]['HAL']

                    curr_time_stamp = df.loc[index]['Date']
                    next_time_stamp = df.loc[index + 1]['Date']
                    
                    curr_24hr_heparin = df.loc[index]['HEPARIN_DOSE']
                    next_24hr_heparin = df.loc[index + 1]['HEPARIN_DOSE']
                    
                    time_diff = hours_between(next_time_stamp, curr_time_stamp)
                    HAL_diff = next_HAL - curr_HAL
                    aPTT_diff = next_aPTT - curr_aPTT
                    
                    for col in columns_to_impute:
                        if not pd.isnull(df.loc[index, col]):
                            df.loc[index, col + '_t_since_last'] = 0.
                            curr_val_is_missing[col] = False
                        else:
                            curr_val_is_missing[col] = True
                    
                    if time_diff >= min_obs_diff and time_diff <= max_obs_diff:
                        
                        # NOTE: We're dropping all of the first entries because of 
                        # the way heparin is being recorded. Specifically, the HEPARIN_DOSE
                        # is actually the cumulative heparin over the last 24 hours, so all 
                        # of the first entries would be trivially "increase heparin actions"
                        # (because the patient would go from no heparin to heparin). 
                        
                        if obs_indx == 1:  # If it's the first observation for patient trajectory
                            act = -99  # Coded action for the first observation in a patient's trajectory
                            time = time_diff  
                            df.loc[index, 'time'] = 0.  
                            df.loc[index + 1, 'time'] = time  
                        
                        elif obs_indx > 1:  # If it's the same 
                            act = self.get_action(next_24hr_heparin, curr_24hr_heparin)
                            time += time_diff
                            df.loc[index + 1, 'time'] = time
                            self.impute_using_last_value_carried_forward(df, index, curr_val_is_missing)
                        
                        else:
                            raise ValueError(
                                'Invalid observation index: obs_indx = {}'.format(obs_indx)
                            )
                        
                        df.loc[index, 'action'] = act
                        obs_indx += 1
                    
                    else:  # Otherwise, if it indicates a different trajectory from the same patient...
                        act = 99  # Coded action for the last observation of a patient's trajectory
                        df.loc[index, 'action'] = act
                        df.loc[index, 'time'] = time
                        if df.loc[index]['TRAJ_ID'] == df.loc[index - 1]['TRAJ_ID']:
                            self.impute_using_last_value_carried_forward(df, index, curr_val_is_missing)
                        
                        time = 0.
                        traj_id += 1
                        traj_indx += 1
                        obs_indx = 1
                
                else:  # Otherwise, if the next observation is not even from the same patient...
                    act = 99  # Coded action for last observation of a patient's trajectory
                    df.loc[index, 'action'] = act
                    df.loc[index, 'time'] = time
                    time = 0.
                    if df.loc[index]['TRAJ_ID'] == df.loc[index - 1]['TRAJ_ID']:
                        self.impute_using_last_value_carried_forward(df, index, curr_val_is_missing)
                    
                    traj_id += 1
                    traj_indx = 1
                    obs_indx = 1
                    
        # We just use binary missingness indicators for now...
        for col in columns_to_impute:
            df[col + '_t_since_last'] = (df[col + '_t_since_last'] > 0) * 1
        return df
    
    
    def impute_using_last_value_carried_forward(self, df, index, current_missing_values):
        for col in [k for k, v in current_missing_values.items() if v]:
            df.loc[index, col] = df.loc[index - 1, col]  # last-value carried forward imputation
            tmp_col = col + '_t_since_last'
            time_since_last_obs = df.loc[index, 'time'] - df.loc[index - 1, 'time']
            df.loc[index, tmp_col] = df.loc[index - 1, tmp_col] + time_since_last_obs
        return df
    
    
    