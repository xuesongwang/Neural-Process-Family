import numpy as np
import collections
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# The NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tesor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration


NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "y_target", "num_total_points", "num_context_points"))



def get_NIFTY_df(directory = '/share/scratch/xuesongwang/metadata/Stock Market Data'):
    filepathlist = os.listdir(directory)
    train_df = []
    val_df = []
    test_df = []

    for file in filepathlist:
        # print("file name:%s"%file)
        if file in ['NIFTY50_all.csv', 'stock_metadata.csv',]:
            continue
        df = pd.read_csv(os.path.join(directory,file))
        df.set_index("Date", drop=False, inplace=True)
        df = df['VWAP']
        # mean = df[:'2016-11-28'].mean()
        # std = df[:'2016-11-28'].std()
        # train_df.append(normalize(df[:'2016-11-28'], mean, std))
        # val_df.append(normalize(df['2016-11-28': '2017-11-29'], mean, std))
        # test_df.append(normalize(df['2017-11-29':], mean, std))
        train_df.append(df[:'2016-11-28'])
        val_df.append(df['2016-11-28': '2017-11-29'])
        test_df.append(df['2017-11-29':])
    print("load NIFTY50 success!")
    return train_df, val_df, test_df

def collate_fns(max_num_context, max_num_extra_target, device):
    def normalize(y):  # normalize on stock-scale
        mean= torch.mean(y,dim=[1, 2], keepdim=True)
        std = torch.std(y,dim=[1, 2], keepdim=True)
        y_norm = (y - mean)/std
        return y_norm

    def collate_fn(batch):
        # Collate
        x = np.stack([x for x, y in batch], 0)
        y = np.stack([y for x, y in batch], 0)

        # Sample a subset of random size
        num_context = np.random.randint(3, max_num_context)
        num_extra_target = np.random.randint(3, max_num_extra_target)

        x = torch.from_numpy(x).float().unsqueeze(-1)
        y = torch.from_numpy(y).float().unsqueeze(-1)
        y = normalize(y)

        inds = np.random.choice(range(x.shape[1]), size=(num_context + num_extra_target), replace=False)
        context_x = x[:, inds][:, :num_context]
        context_y = y[:, inds][:, :num_context]

        target_x = x[:, inds][:, num_context:]
        target_y = y[:, inds][:, num_context:]


        query = ((context_x, context_y), target_x)
        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points=num_extra_target + num_context,
            num_context_points=num_context)
    return collate_fn


class NIFTY(Dataset):
    def __init__(self, df, max_length = 365):
        self.df = df
        self.max_length = max_length

    def get_rows(self, i):
        stock = self.df[i]
        try:
            index = np.random.randint(stock.shape[0] - self.max_length)
        except:
            print("stock length:",stock.shape[0])
        slice = stock.iloc[index: index + self.max_length].copy()
        x = (pd.to_datetime(slice.index) -  pd.to_datetime(slice.index[0])).days/30 # derive day difference, then normalize by month
        return x, slice

    def __getitem__(self, i):
        x, y = self.get_rows(i)
        return x.values, y.values

    def __len__(self):
        return len(self.df)

class NIFTYReader:
    def __init__(self, batch_size = 50,max_num_context = 50,  device = torch.device("cpu")):
        super().__init__()
        self._dfs = None
        self.num_context = max_num_context
        self.num_extra_target = max_num_context
        self.batch_size = batch_size
        self.num_workers = 4
        self.device = device

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_val, df_test = get_NIFTY_df()
            self._dfs = dict(df_train=df_train, df_val = df_val, df_test=df_test)
        return self._dfs

    def train_dataloader(self):
        df_train = self._get_cache_dfs()['df_train']
        data_train = NIFTY(df_train)
        return torch.utils.data.DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fns(self.num_context, self.num_extra_target, self.device),
            num_workers=self.num_workers)

    def val_dataloader(self):
        df_val = self._get_cache_dfs()['df_val']
        data_val = NIFTY(df_val, max_length=250)
        return torch.utils.data.DataLoader(
            data_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fns(self.num_context, self.num_extra_target, self.device),
        )

    def test_dataloader(self):
        df_test = self._get_cache_dfs()['df_test']
        data_test = NIFTY(df_test)
        return torch.utils.data.DataLoader(
            data_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fns(
                self.num_context, self.num_extra_target,  self.device),
        )

if __name__ == '__main__':
    dataset = NIFTYReader()
    train_loader = dataset.train_dataloader()
    for i, batch in enumerate(train_loader):
        batch
        print("pause")