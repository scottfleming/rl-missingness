import numpy as np
from sklearn.model_selection import KFold

def generate_k_patient_id_folds(unique_patient_ids, k, seed=42):
    """Randomly generate K disjoint train/test folds of (unique) patient IDs
    """
    assert len(np.unique(unique_patient_ids)) == len(unique_patient_ids)
    np.random.seed(seed)
    # np.random.shuffle(unique_patient_ids)
    kf = KFold(n_splits=k, shuffle=True)
    patient_ids = {}
    for i, (train_indxs, test_indxs) in enumerate(kf.split(unique_patient_ids)):
        fold_name = 'fold' + str(i + 1)
        patient_ids[fold_name] = {}
        patient_ids[fold_name]['train'] = unique_patient_ids[train_indxs]
        patient_ids[fold_name]['test'] = unique_patient_ids[test_indxs]
    return patient_ids