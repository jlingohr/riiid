import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader


def build_user_series(df, max_seq_len):
    """
    Split each users' sequence into sequences of at most max_seq_len.
    Start from end index and build backwards
    """
    user_ids, sequence_num, row_mins, row_maxs = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_min = row.row_min
        row_max = row.row_max
        length = row_max - row_min + 1
        num_sequences, final_seq_length = divmod(length, max_seq_len)

        if final_seq_length > 0:
            user_ids.append(row.user_id)
            sequence_num.append(0)
            row_mins.append(row_min)
            row_maxs.append(row_max - (num_sequences * max_seq_len) - num_sequences)
        for i in reversed(range(0, num_sequences)):
            user_ids.append(row.user_id)
            sequence_num.append(i+1)
            row_maxs.append(row_max - (i * max_seq_len) - i)
            row_mins.append(row_max - i - ((i+1) * max_seq_len))

    df = pd.DataFrame({'user_ids' : user_ids, 'sequence_nums' : sequence_num, 'row_min' : row_mins, 'row_max' : row_maxs})
    return df


def build_windowed_series(df, window_size=528, overlap=0.125):
    """
    Split user sequence into overlapping subsequences from which to sample
    """
    step_size = int(window_size - (window_size * overlap))
    user_ids, sequence_num, row_mins, row_maxs = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_min = row.row_min
        row_max = row.row_max
        length = row_max - row_min + 1

        # If current user sequence is less than window_size, create single row
        if length < window_size:
            user_ids.append(row.user_id)
            sequence_num.append(0)
            row_mins.append(row_min)
            row_maxs.append(row_max)
        else:
            x_steps = int(ceil((length - window_size) / step_size)) + 1
            sequence = np.arange(row_min, row_max+1)
            for idx in range(x_steps):
                x_idx = int(min(length - window_size, (idx % x_steps) * step_size))
                subseq = sequence[x_idx : x_idx + window_size]
                user_ids.append(row.user_id)
                sequence_num.append(idx)
                row_mins.append(subseq[0])
                row_maxs.append(subseq[-1])

    df = pd.DataFrame({'user_ids' : user_ids, 'sequence_nums' : sequence_num, 'row_min' : row_mins, 'row_max' : row_maxs})
    return df


def get_min_max(df):
    grouped = df.loc[:, ['user_id', 'row_id']].groupby(['user_id'])
    grouped = grouped.row_id.agg(['min', 'max']).reset_index()
    grouped.rename(columns={'min': 'row_min', 'max' : 'row_max'}, inplace=True)
    return grouped


def load_data(fold_idx, max_seq_len, train_root, debug=False):
    dtypes = {
        "row_id": "int64",
        "timestamp": "int64",
        "user_id": "int32",
        "content_id": "int16",
        "content_type_id": "boolean",
        "task_container_id": "int16",
        "user_answer": "int8",
        "answered_correctly": "int8",
        "prior_question_elapsed_time": "float32",
        "prior_question_had_explanation": "boolean"
    }
    # Load main data files
    usecols = ['timestamp', 'user_id', 'content_id', 'answered_correctly', 'prior_question_elapsed_time']
    data = pd.read_feather(os.path.join(train_root, 'train.feather'))
    data = data[data.content_type_id == False]
    data = data[usecols]
    data.reset_index(drop=True, inplace=True)
    data['row_id'] = data.index

    # Add lagtime - should discretize ms into minutes, 0, 1, 2, 3, 4, 5, 10, 20, 30, ..., 1440, so 150 total
    data['lagtime'] = data.groupby('user_id')['timestamp'].shift()
    data['lagtime']=data['timestamp']-data['lagtime']
    data['lagtime'].fillna(0, inplace=True)
    data.lagtime = ((data.lagtime/(1000*60))).astype('int32')
    data.lagtime[data.lagtime > 1440] = 1440
    data.lagtime[data.lagtime > 5] = (data.lagtime[data.lagtime > 5] / 10 + 5).astype('int32')

    # Add attempt number
    data["attempt_no"] = 1
    data.attempt_no=data.attempt_no.astype('int8')
    data["attempt_no"] = data[["user_id","content_id",'attempt_no']].groupby(["user_id","content_id"])["attempt_no"].cumsum()
    data['has_attempted'] = (data.attempt_no > 1).astype('bool')

    # Load row indices to use in train and val
    train_rows = pd.read_feather(os.path.join(train_root, 'fold_{}'.format(fold_idx), 'train_rows.feather'))
    val_rows = pd.read_feather(os.path.join(train_root, 'fold_{}'.format(fold_idx), 'val_rows.feather'))

    # Filter out by rows
    train = data[data.index.isin(train_rows.row_id)]
    train.reset_index(drop=True, inplace=True)
    train['row_id'] = train.index
    val = data[data.index.isin(val_rows.row_id)]
    val.reset_index(drop=True, inplace=True)
    val['row_id'] = val.index
    print("{} training examples loaded".format(len(train)))
    print("{} validation examples loaded".format(len(val)))


    # prepare user_id groups and partition in window_sized sequences

    if os.path.exists(os.path.join(train_root, 'train_group_0.csv')):
        train_users = pd.read_csv(os.path.join(train_root, 'train_group_{}.csv'.format(fold_idx)))
        val_users = pd.read_csv(os.path.join(train_root, 'val_group_{}.csv'.format(fold_idx)))
    else:
        train_users = get_min_max(train)
        val_users = get_min_max(val)
        if debug:
            train_users = train_users.loc[: int(0.1 * len(train_users)), :]
            val_users = val_users.loc[: int(0.1 * len(val_users)), :]

        train_users = build_windowed_series(train_users)
        val_users = build_user_series(val_users, max_seq_len)

        train_users.to_csv(os.path.join(train_root, 'train_group_{}.csv'.format(fold_idx)), index=False)
        val_users.to_csv(os.path.join(train_root, 'val_group_{}.csv'.format(fold_idx)), index=False)
    print("{} training users loaded".format(len(train_users)))
    print("{} validation users loaded".format(len(val_users)))

    # Create dataset objects
    train = Riiid(train, train_users, max_seq_len)
    val = Riiid(val, val_users, max_seq_len)

    return train, val


class Riiid(Dataset):
    def __init__(self, df, users, max_seq_length):
        self.user_ids = df.loc[:, 'user_id'].to_numpy()
        self.questions = df.loc[:, 'content_id'].to_numpy()
        self.prior_elapsed_time = df.loc[:, 'prior_question_elapsed_time'].fillna(0).to_numpy() / 300000.0
        self.lagtime = df.loc[:, 'lagtime'].fillna(0).to_numpy()
        self.has_attempted = df.loc[:, 'has_attempted'].fillna(False).astype('int').to_numpy()
        self.target = df.loc[:, 'answered_correctly'].to_numpy()
        self.users = users
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        row = self.users.iloc[idx]
        row_min, row_max = row.row_min, row.row_max

        # If subsequence longer than max_seq_length, subsample a window
        if row_max - row_min >= self.max_seq_length:
            row_min = np.random.randint(row_min, row_max - self.max_seq_length + 1)
            row_max = row_min + self.max_seq_length - 1

        user_ids = self.user_ids[row_min : row_max + 1]
        question_ids = self.questions[row_min : row_max+1]# + 1
        prior_question_elapsed_time = self.prior_elapsed_time[row_min : row_max+1]
        lagtime = self.lagtime[row_min : row_max+1]
        has_attempted = self.has_attempted[row_min : row_max+1]
        target = self.target[row_min : row_max+1]
        padded = np.array([False] * len(question_ids) + [True] * (self.max_seq_length - len(question_ids))) #TODO is this correct?

        # pad everything to max_seq_length
        pad_width = self.max_seq_length - len(question_ids)
        if pad_width > 0:
            user_ids = np.pad(user_ids, (0, pad_width))
            question_ids = np.pad(question_ids, (0, pad_width))
            prior_question_elapsed_time = np.pad(prior_question_elapsed_time, (0, pad_width))
            lagtime = np.pad(lagtime, (0, pad_width))
            has_attempted = np.pad(has_attempted, (0, pad_width))
            target = np.pad(target, (0, pad_width), mode='constant', constant_values=(2,2))

        sample = {
            'user_ids' : user_ids,
            'question_ids' : question_ids,
            'prior_elapsed_time' : prior_question_elapsed_time,
            'lagtime' : lagtime,
            'has_attempted' : has_attempted,
            'padded': padded,
            'target' : target
            }

        return sample
