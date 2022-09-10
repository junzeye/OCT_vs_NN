import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CTGdataset(Dataset):

    def __init__(self, _class:int = 10):

        df = pd.read_csv('./data/CTG/CTG.csv', skiprows = 1, header=None, delimiter=',')
        df.replace('', np.nan, inplace=True)
        df.dropna(inplace = True)
        if _class == 3:
            x, y = df[[i for i in range(22)]], df[23].astype(int)
        else: # i.e. _class == 10
            x, y = df[[i for i in range(22)]], df[22].astype(int)
        1+1
        self.x=torch.tensor(x.values, dtype=torch.float32)
        self.y=torch.tensor(y.values, dtype=torch.long) - 1
        # -1 to make sure all class labels start with 0 and end with k-1
 
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':
    whole_dataset = CTGdataset()
    batch_size = 10
    train_dataset, test_dataset = torch.utils.data.random_split(whole_dataset, 
        [int(0.5 * len(whole_dataset)), len(whole_dataset) - int(0.5 * len(whole_dataset))])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)

    train_iter = iter(train_loader)

    features, labels = train_iter.next()

    print('features shape on batch size = {}'.format(features.size()))
    print('labels shape on batch size = {}'.format(labels.size()))

    print("")