import xarray as xr
import torch
from torch import nn
from torch.utils.data import Dataset
import os.path


def relative_MSELoss(output, target):
    loss = torch.mean((output - target)**2) / torch.mean(target**2)
    return loss


class QBODataset(Dataset):
    def __init__(self, file_path, feature_transform=None, label_transform=None):
        if os.path.isfile(file_path):
            ds = xr.open_dataset(file_path)

        ti = 0
        te = 360*96

        self.features = torch.tensor(ds.u.values[ti:te, :])
        self.labels = torch.tensor(ds.S.values[ti:te, :])

        self.feature_transform = feature_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        feature = self.features[idx, :]
        label = self.labels[idx, :]

        if self.feature_transform:
            feature = self.feature_transform(feature)
        if self.label_transform:
            label = self.label_transform(label)

        return feature, label


class GlobalStandardScaler():
    def __init__(self, X):
        self.mean = X.mean()
        self.std = X.std()

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean


class GlobalMaxScaler():
    def __init__(self, X):
        self.abs_max = X.abs().max()
        
    def transform(self, X):
        return X / self.abs_max
    
    def inverse_transform(self, X):
        return X * self.abs_max
    

class FullyConnected(nn.Module):
    def __init__(self, solver, scaler_X=None, scaler_Y=None):
        super(FullyConnected, self).__init__()

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        nlev = solver.z.shape[0]
        
        self.linear_relu_stack = nn.Sequential(            
            nn.Linear(1*(nlev-2), 3*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(3*(nlev-2), 6*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(6*(nlev-2), 9*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(9*(nlev-2), 9*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(9*(nlev-2), 9*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(9*(nlev-2), 9*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(9*(nlev-2), 9*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(9*(nlev-2), 6*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(6*(nlev-2), 3*(nlev-2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(3*(nlev-2), 1*(nlev-2), dtype=torch.float64)
        )

    def forward(self, X):
        if self.scaler_X:
            X = self.scaler_X.transform(X)

        if self.training:
            Y = self.linear_relu_stack(X[:, 1:-1])
        else:
            Y = self.linear_relu_stack(X[1:-1])
            
        if not self.training:
            if self.scaler_Y:
                Y = self.scaler_Y.inverse_transform(Y)
            Y = torch.hstack((torch.zeros(1), Y, torch.zeros(1)))

        return Y

