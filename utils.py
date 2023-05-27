from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import string
import os
import torch

mask = ''.join(random.sample(string.ascii_letters, 8))


def set_gpu(algo):
    if torch.cuda.is_available():
        device = 'cuda'
        if algo == 'devnet' or algo == 'feawad' or algo == 'prenet' :
            print('using tf')
            import keras.backend.tensorflow_backend as KTF
            import tensorflow as tf
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            conf = tf.ConfigProto()
            conf.gpu_options.allow_growth = True
            sess = tf.Session(config=conf)
            KTF.set_session(sess)
        # elif algo == 'des' or algo == 'rosas':
        #     torch.cuda.set_device(0)
    else:
        device = 'cpu'
    return device


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=1e-4,
                 model_name="", trace_func=print, structrue='torch'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.structure = structrue

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.trace_func = trace_func

        if structrue == 'torch':
            self.path = "checkpoints/" + model_name + "." + mask + '_checkpoint.pt'
        elif structrue == 'keras':
            self.path = "checkpoints/" + model_name + '.' + mask + "_checkpoint.h5"
        if not os.path.exists(os.path.split(self.path)[0]):
            os.mkdir(os.path.split(self.path)[0])

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.structure == 'torch':
            torch.save(model.state_dict(), self.path)
        elif self.structure == 'keras':
            model.save(self.path)

        self.val_loss_min = val_loss


def evaluate(y_true, y_prob):
    auroc = metrics.roc_auc_score(y_true, y_prob)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall, precision)
    return auroc, aupr


def split_train_test(x, y, test_size, random_state=None):
    idx_norm = y == 0
    idx_out = y == 1

    n_f = x.shape[1]
    del_list = []
    for i in range(n_f):
        if np.std(x[:, i]) == 0:
            del_list.append(i)
    if len(del_list) > 0:
        print("Pre-process: Delete %d features as every instances have the same behaviour: " % len(del_list))
        x = np.delete(x, del_list, axis=1)

    # keep outlier ratio, norm is normal out is outlier
    x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(x[idx_norm], y[idx_norm],
                                                                            test_size=test_size,
                                                                            random_state=random_state)
    x_train_out, x_test_out, y_train_out, y_test_out = train_test_split(x[idx_out], y[idx_out],
                                                                        test_size=test_size,
                                                                        random_state=random_state)
    x_train = np.concatenate((x_train_norm, x_train_out))
    x_test = np.concatenate((x_test_norm, x_test_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))

    # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
    # scaler = StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x_train)
    x_train = minmax_scaler.transform(x_train)
    x_test = minmax_scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def split_train_test_val(x, y, test_ratio, val_ratio, random_state=None, del_features=True):
    idx_norm = y == 0
    idx_out = y == 1

    n_f = x.shape[1]

    if del_features:
        del_list = []
        for i in range(n_f):
            if np.std(x[:, i]) == 0:
                del_list.append(i)
        if len(del_list) > 0:
            print("Pre-process: Delete %d features as every instances have the same behaviour: " % len(del_list))
            x = np.delete(x, del_list, axis=1)

    # from sklearn.model_selection import train_test_split
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
    #                                                     random_state=2, stratify=y)
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5,
    #                                                 random_state=2, stratify=y_test)
    # print('train/val/test size:', len(x_train), len(x_val), len(x_test))
    # from collections import Counter
    # print(Counter(y_train))
    # print(Counter(y_val))
    # print(Counter(y_test))

    # keep outlier ratio, norm is normal out is outlier
    x_train_norm, x_teval_norm, y_train_norm, y_teval_norm = train_test_split(x[idx_norm], y[idx_norm],
                                                                              test_size=test_ratio + val_ratio,
                                                                              random_state=random_state)
    x_train_out, x_teval_out, y_train_out, y_teval_out = train_test_split(x[idx_out], y[idx_out],
                                                                          test_size=test_ratio + val_ratio,
                                                                          random_state=random_state)

    x_test_norm, x_val_norm, y_test_norm, y_val_norm = train_test_split(x_teval_norm, y_teval_norm,
                                                                        test_size=val_ratio / (test_ratio + val_ratio),
                                                                        random_state=random_state)
    x_test_out, x_val_out, y_test_out, y_val_out = train_test_split(x_teval_out, y_teval_out,
                                                                    test_size=val_ratio / (test_ratio + val_ratio),
                                                                    random_state=random_state)

    x_train = np.concatenate((x_train_norm, x_train_out))
    x_test = np.concatenate((x_test_norm, x_test_out))
    x_val = np.concatenate((x_val_norm, x_val_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))
    y_val = np.concatenate((y_val_norm, y_val_out))

    from collections import Counter
    print('train counter', Counter(y_train))
    print('val counter  ', Counter(y_val))
    print('test counter ', Counter(y_test))

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x_train)
    x_train = minmax_scaler.transform(x_train)
    x_test = minmax_scaler.transform(x_test)
    x_val = minmax_scaler.transform(x_val)

    return x_train, y_train, x_test, y_test, x_val, y_val


def semi_setting(y_train, n_known_outliers=30):
    """
    default: using ratio to get known outliers, also can using n_known_outliers to get semi-y
    use the first k outlier as known
    :param y_train:
    :param n_known_outliers:
    :return:
    """
    outlier_indices = np.where(y_train == 1)[0]
    n_outliers = len(outlier_indices)
    n_known_outliers = min(n_known_outliers, n_outliers)

    # rng = np.random.RandomState(random_state)
    # known_idx = rng.choice(outlier_indices, n_known_outliers, replace=False)
    known_idx = outlier_indices[:n_known_outliers]

    new_y_train = np.zeros_like(y_train, dtype=int)
    new_y_train[known_idx] = 1
    return new_y_train


def semi_setting2(y_train, n_known_outliers=30, random_state=42):
    """
    default: using ratio to get known outliers, also can using n_known_outliers to get semi-y
    use the first k outlier as known
    :param y_train:
    :param n_known_outliers:
    :return:
    """
    outlier_indices = np.where(y_train == 1)[0]
    n_outliers = len(outlier_indices)
    n_known_outliers = min(n_known_outliers, n_outliers)

    rng = np.random.RandomState(random_state)
    known_idx = rng.choice(outlier_indices, n_known_outliers, replace=False)

    new_y_train = np.zeros_like(y_train, dtype=int)
    new_y_train[known_idx] = 1
    return new_y_train



def filter_dataset(data_path, min_anom=70, min_obj=600):
    df = pd.read_csv(data_path)
    is_use = True
    n_obj = df.shape[0]
    y = np.array(df.values[:, -1], dtype=int)
    n_anom = len(np.where(y == 1)[0])

    if n_obj < min_obj:
        is_use = False
    if n_anom < min_anom:
        is_use = False

    return is_use


def min_max_norm(array):
    array = np.array(array)
    _min_, _max_ = np.min(array), np.max(array)
    if _min_ == _max_:
        raise ValueError("Given a array with same max and min value in normalisation")
    norm_array = np.array([(a - _min_) / (_max_ - _min_) for a in array])
    return norm_array


def filter_noise(x_train, y_train, semi_y, remove_ratio=0.1):
    known_anom_idx = np.where(semi_y == 1)[0]
    true_anom_idx = np.where(y_train == 1)[0]
    unknown_anom_idx = np.setdiff1d(true_anom_idx, known_anom_idx)
    n_noise = len(unknown_anom_idx)
    n_remove = int(np.ceil(remove_ratio * n_noise))

    remove_id = unknown_anom_idx[np.random.choice(n_noise, n_remove, replace=False)]

    x_train = np.delete(x_train, remove_id, 0)
    y_train = np.delete(y_train, remove_id, 0)
    semi_y = np.delete(semi_y, remove_id, 0)

    return x_train, y_train, semi_y


def adjust_contamination(x_train, y_train, semi_y, adjust_cont_r, random_state):
    """
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swap 5% features of two anomalies to avoid duplicate contaminated anomalies.
    """
    rng = np.random.RandomState(random_state)

    known_anom_idx = np.where(semi_y == 1)[0]
    true_anom_idx = np.where(y_train == 1)[0]
    true_anoms = x_train[true_anom_idx]
    unknown_anom_idx = np.setdiff1d(true_anom_idx, known_anom_idx)

    n_adj_noise = int(len(np.where(y_train == 0)[0]) * adjust_cont_r / (1. - adjust_cont_r))
    n_cur_noise = len(unknown_anom_idx)

    # x_train = np.delete(x_train, unknown_anom_idx, axis=0)
    # y_train = np.delete(y_train, unknown_anom_idx, axis=0)
    # noises = inject_noise(true_anoms, n_adj_noise, 42)
    # x_train = np.append(x_train, noises, axis=0)
    # y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    # inject noise
    if n_cur_noise < n_adj_noise:
        print('Control Contamination Rate: Injecting Noise')
        n_inj_noise = n_adj_noise - n_cur_noise
        if len(unknown_anom_idx) > 2:
            seed_anomalies = x_train[unknown_anom_idx]
        else:
            seed_anomalies = x_train[known_anom_idx]
        n_sample, dim = seed_anomalies.shape
        swap_ratio = 0.05
        n_swap_feat = int(swap_ratio * dim)
        inj_noise = np.empty((n_inj_noise, dim))
        for i in np.arange(n_inj_noise):
            idx = rng.choice(n_sample, 2, replace=False)
            o1 = seed_anomalies[idx[0]]
            o2 = seed_anomalies[idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            inj_noise[i] = o1.copy()
            inj_noise[i, swap_feats] = o2[swap_feats]

        x_train = np.append(x_train, inj_noise, axis=0)
        y_train = np.append(y_train, np.ones(n_inj_noise))
        semi_y = np.append(semi_y, np.zeros(n_inj_noise))

    # remove noise
    elif n_cur_noise > n_adj_noise:
        print('Control Contamination Rate: Removing Noise')
        n_remove = n_cur_noise - n_adj_noise
        remove_id = unknown_anom_idx[rng.choice(n_cur_noise, n_remove, replace=False)]

        x_train = np.delete(x_train, remove_id, 0)
        y_train = np.delete(y_train, remove_id, 0)
        semi_y = np.delete(semi_y, remove_id, 0)

    return x_train, y_train, semi_y


def inject_noise(seed, n_out, random_seed):
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise