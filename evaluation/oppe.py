import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm

def train_propensity_models(df, 
                            propensity_model_features, 
                            actions_actually_taken='action'):
    # propensity model
    pi_model = CalibratedClassifierCV(
        base_estimator=RandomForestClassifier(class_weight='balanced', 
                                              n_estimators=100),
        method='isotonic',
        cv=5
    )
    
    pi_model.fit(
        df[propensity_model_features], 
        df[actions_actually_taken].astype(int)
    )
    # pickle.dump(pi_model, open("ckpts/pi_model.pkl", "wb"))
    return pi_model
    
    
def get_importance_sampling_returns(eval_df, 
                                    behavior_policy_col, 
                                    evaluation_policy_col, 
                                    propensity_model_features,
                                    reward_col,
                                    trajectory_id_col,
                                    discount_factor=0.9):
    
    # Learn a model of the data-generating behavior policy
    pi_b_model = train_propensity_models(eval_df,
                                         propensity_model_features=propensity_model_features,
                                         actions_actually_taken=behavior_policy_col)
    
    # Use the propensity model to generate predicted probabilities for each action that
    # was taken in the historical data, given the state at the time.
    # In other words, under our propensity model, how likely was it for the 
    # behavior agent to take the action that they took?
    pi_b_preds = pi_b_model.predict_proba(eval_df[propensity_model_features])
    
    # reward_models = train_reward_models(eval_df,
    #                                     reward_model_features=reward_model_features,
    #                                     actions_actually_taken=behavior_policy_col,
    #                                     reward_col=reward_col)
    
    # Use the rewards models to generate predicted rewards for each possible action
    # at each state observed in the historical data
    # reward_preds = np.zeros((eval_df.shape[0], len(reward_models)))
    # for action, model in reward_models.items():
    #     reward_preds[:, int(action)] = model.predict(eval_df[reward_model_features])
    
    # A matrix M where each row i represents an observation, each column j represents
    # an action encoded as an integer, and each element M[i, j] = 1 if the agent
    # actually took action j at observation i in the historical data and 0 otherwise
    pi_b_mask = np.c_[eval_df[behavior_policy_col].astype(int).values == 0,
                      eval_df[behavior_policy_col].astype(int).values == 1,
                      eval_df[behavior_policy_col].astype(int).values == 2,
                      eval_df[behavior_policy_col].astype(int).values == 3]
    
    # A matrix M where each row i represents an observation, each column j represents
    # an action encoded as an integer, and each element M[i, j] = 1 if our evaluation
    # policy recommends taking action j at observation i in the historical data and
    # 0 otherwise
    pi_e_mask = np.c_[eval_df[evaluation_policy_col].astype(int).values == 0,
                      eval_df[evaluation_policy_col].astype(int).values == 1,
                      eval_df[evaluation_policy_col].astype(int).values == 2,
                      eval_df[evaluation_policy_col].astype(int).values == 3]
    
    # All of the rows should sum to 1 in these matrices because exactly one action
    # should have been taken/recommended by the behavior agent/evaluation policy, 
    # respectively
    assert np.all(np.sum(pi_b_mask, axis=1) == 1)
    assert np.all(np.sum(pi_e_mask, axis=1) == 1)
   
    # The propensity for each action taken under the behavior policy, given the state, is 
    # in this case our approximate model's predicted probability of the behavior agent 
    # taking the action that was, in fact, taken in the historical data/trajectory
    pi_b = np.sum(pi_b_preds * pi_b_mask, axis=1)
    
    if evaluation_policy_col == behavior_policy_col:
        # TODO: Just let the logic handle the case where you want the clinician policy?
        pi_e = pi_b
    else:
        # Assuming our evaluation policy is deterministic, the propensity under the 
        # evaluation policy for each action taken in the data is 1 for those cases 
        # in which the evaluation policy matches the behavior policy (i.e. 1 if the behavior
        # agent took the action that our evaluation policy would have recommended) and 0
        # otherwise (i.e. 0 if the agent took an action other than the action that our 
        # evaluation policy would have recommended)
        pi_e = (eval_df[evaluation_policy_col] == eval_df[behavior_policy_col]).astype(float)
        pi_e = pi_e.values    
    
    eval_df['pi_e'] = pi_e
    eval_df['pi_b'] = pi_b
    eval_df['weights'] = pi_e / pi_b
    
    # See table 1 of https://realworld-sdm.github.io/paper/34.pdf for 
    # a description of the algorithms implemented below.
    # In this section of code we use the propensity score models trained above 
    # in order to calculate the running product of ratios of propensity weights
    # i.e. rho^i_{0:t} = prod_{t=0}^{min(t, H_i)} (pi_e(a^i_t | s^i_t) / pi_b(a^i_t | s^i_t))
    # and w_{0:t} = sum_{i=1}^|D| (rho^i_{0:t}) where H_i is the horizon of trajectory i.
    # We do this by creating a dataframe wherein each column i represents a trajectory and each row t
    # represents a step in the horizon such that rho_df[t, i] = rho^i_{0:t} and
    # w_per_step[t] = w_{0:t}, using the same notation above
    max_traj_len = np.max(eval_df['obs_num'])
    rho = {}
    for traj_id in np.unique(eval_df[trajectory_id_col]):
        traj_id = int(traj_id)
        pdf = eval_df.loc[eval_df[trajectory_id_col] == traj_id]
        rho[traj_id] = []
        running_rho = 1.
        for indx in pdf.index:
            running_rho *= pdf.loc[indx, 'weights']
            rho[traj_id].append(running_rho)
        
        # In order to get proper estimates of w_{0:t} we need rho^i_{0:t} for all
        # trajectories i = 1, 2, ..., |D| in the dataset. But note that each of those
        # rho^i_{0:t} = prod_{t=0}^{min(t, H_i)} [...] are "capped" at the length of the 
        # horizon. So, to make summing across rows meaningful when trajectories have mixed
        # horizon lengths, we simply extend the estimates of rho^i_{0:t} to 
        # H_max, the maximum length trajectory in the dataset, by propagating rho^i_{0:H_i} forward
        # to fill out the rest of the array of length H_max
        while(len(rho[traj_id]) < max_traj_len):
            rho[traj_id].append(running_rho)
            
    rho_df = pd.DataFrame(rho)
    w_per_step = dict(np.sum(rho_df, axis=1))
    
    # The trajectory-specific product of propensity score ratios over
    # the horizon of the trajectory, i.e. rho^i_{0:H_i}
    rho_H = eval_df.groupby(trajectory_id_col).apply(lambda x: np.prod(x['weights']))
    rho_H.index = np.array(rho_H.index, dtype=np.int)
    
    num_trajs = len(np.unique(eval_df[trajectory_id_col]))
    
    standard_IS_patient_value = {}
    standard_WIS_patient_value = {}
    stepwise_IS_patient_value = {}
    stepwise_WIS_patient_value = {}
    
    for traj_id in np.unique(eval_df[trajectory_id_col]):
        traj_id = int(traj_id)
        pdf = eval_df.loc[eval_df[trajectory_id_col] == traj_id]
        H_i = len(pdf) - 1  # Horizon index for trajectory traj_id
        
        standard_IS_patient_value[traj_id] = 0.
        standard_WIS_patient_value[traj_id] = 0.
        stepwise_IS_patient_value[traj_id] = 0.
        stepwise_WIS_patient_value[traj_id] = 0.
        
        rho_t = 1.
        discount_t = 1.
        for t, indx in enumerate(pdf.index):
            rho_t = rho_t * pdf.loc[indx, 'weights']
            reward_t = pdf.loc[indx, reward_col]
            standard_IS_patient_value[traj_id] += discount_t * (rho_H[traj_id] / num_trajs) * reward_t
            standard_WIS_patient_value[traj_id] += discount_t * (rho_H[traj_id] / w_per_step[H_i]) * reward_t
            stepwise_IS_patient_value[traj_id] += discount_t * (rho_t / num_trajs) * reward_t
            stepwise_WIS_patient_value[traj_id] += discount_t * (rho_t / w_per_step[t]) * reward_t
            discount_t = discount_t * discount_t
    
    standard_IS_scores = np.array([v for k,v in standard_IS_patient_value.items()])
    standard_WIS_scores = np.array([v for k,v in standard_WIS_patient_value.items()])
    stepwise_IS_scores = np.array([v for k,v in stepwise_IS_patient_value.items()])
    stepwise_WIS_scores = np.array([v for k,v in stepwise_WIS_patient_value.items()])
    
    # print('standard_IS_scores = {}'.format(np.sum(standard_IS_scores)))
    # print('standard_WIS_scores = {}'.format(np.sum(standard_WIS_scores)))
    # print('stepwise_IS_scores = {}'.format(np.sum(stepwise_IS_scores)))
    # print('stepwise_WIS_scores = {}'.format(np.sum(stepwise_WIS_scores)))
    # print('num_trajs = {}'.format(num_trajs))
    return standard_IS_scores, standard_WIS_scores, stepwise_IS_scores, stepwise_WIS_scores, num_trajs