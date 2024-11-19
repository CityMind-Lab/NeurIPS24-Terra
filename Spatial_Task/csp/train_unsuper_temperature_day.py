import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
import pickle
from argparse import ArgumentParser

from torch import optim
import models
import utils as ut
import datasets as dt
import data_utils as dtul
import grid_predictor as grid
from paths import get_paths
import losses as lo

from dataloader import *
from trainer_helper import *
from eval_helper import *
from trainer import *



import sys
# sys.path.append('./satclip')
import matplotlib.pyplot as plt
import torch
# from load import get_satclip
from mpl_toolkits.basemap import Basemap
from urllib import request
import numpy as np
import pandas as pd
import io
import torch
from torch.utils.data import TensorDataset, random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Automatically select device
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super(MLP, self).__init__()

        layers = []
        layers += [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()] # Input layer
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # Hidden layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)] # Output layer

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

def convert_to_lat_lon(data):
    """
    Converts a list of numbers between 0 and 64799 to a list of latitudes and longitudes within the range [-89, 90] and [-180, 179].

    Args:
        data (list): A list of numbers between 0 and 64799.

    Returns:
        list: A list of tuples representing latitudes and longitudes.
    """

    latitudes = []
    longitudes = []

    for number in data:
        # Convert number to latitude and longitude coordinates
        latitude = (number // 360) - 89
        longitude = (number % 360) - 180

        # Append coordinates to lists
        latitudes.append(latitude)
        longitudes.append(longitude)

    return list(zip(latitudes, longitudes))

def calculate_lat_lon(row):
    # 这里是你的函数，我假设它是func
    # import ipdb; ipdb.set_trace()
    lat, lon = convert_to_lat_lon([row.name])[0]
    lat = lat + 0.5
    lon = lon + 0.5
    return pd.Series({'lat': lat, 'lon': lon})

def get_air_temp_data(pred="temp",
                      norm_y=True,
                      norm_x=True
                      ):
    '''
    Download and process the Global Air Temperature dataset (more info: https://www.nature.com/articles/sdata2018246)

    Parameters:
    pred = numeric; outcome variable to be returned; choose from ["temp", "prec"]
    norm_y = logical; should outcome be normalized
    norm_min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

    Return:
    coords = spatial coordinates (lon/lat)
    x = features at location
    y = outcome variable
    '''
    #   url = 'https://springernature.figshare.com/ndownloader/files/12609182'
    #   url_open = request.urlopen(url)
    
    # inc = np.array(pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))))
    inc = pd.read_csv('../../data/temperature_avg_day.csv',names=['temperature'])
    # 定义函数来计算经纬度

    # 计算经纬度并添加到DataFrame
    inc[['lat', 'lon']] = inc.apply(calculate_lat_lon, axis=1)
    inc = np.array(inc)
    y = inc[:,0].reshape(-1)
    coords = inc[:,1:]
    # import ipdb; ipdb.set_trace()
    # coords = coords[:,::-1]
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std

    return torch.tensor(coords.copy()),  torch.tensor(y), y_mean, y_std



#先lon再lat
coords, y, y_mean, y_std = get_air_temp_data()
# print(coords)
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

fig, ax = plt.subplots(1, figsize=(5, 3))

m = Basemap(projection='cyl', resolution='c', ax=ax)
m.drawcoastlines()
scatter = ax.scatter(coords[:,0], coords[:,1], c=y, s=5)
ax.set_title('AirTemperature')

# 添加colorbar
cbar = plt.colorbar(scatter)
# cbar.set_label('wind')

plt.show()
parser = make_args_parser()
args = parser.parse_args()

trainer = Trainer(args, console = True)

# import ipdb; ipdb.set_trace()
# satclip_path = './satclip-vit16-l10.ckpt'
model = trainer.model
# model = get_satclip(satclip_path, device=device) # Only loads location encoder by default
model.eval()
with torch.no_grad():
  x  = model(coords.double().to(device)).detach().cpu()

print(coords.shape)
print(x.shape)


dataset = TensorDataset(coords, x, y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

coords_train, x_train, y_train = train_set.dataset.tensors[0][train_set.indices], train_set.dataset.tensors[1][train_set.indices], train_set.dataset.tensors[2][train_set.indices]
coords_test, x_test, y_test = test_set.dataset.tensors[0][test_set.indices], test_set.dataset.tensors[1][test_set.indices], test_set.dataset.tensors[2][test_set.indices]

fig, ax = plt.subplots(1, figsize=(5, 3))

m = Basemap(projection='cyl', resolution='c', ax=ax)
m.drawcoastlines()
ax.scatter(coords_train[:,0], coords_train[:,1], c='blue', s=2, label='Training',alpha=0.5)
ax.scatter(coords_test[:,0], coords_test[:,1], c='green', s=2, label='Testing',alpha=0.5)
ax.legend()
ax.set_title('Train-Test Split')

# plt.savefig('pra.jpg')
# import ipdb; ipdb.set_trace()
pred_model = MLP(input_dim=8142, dim_hidden=64, num_layers=2, out_dims=1).float().to(device)
criterion = nn.MSELoss() #mae
# criterion = nn.L1Loss() #MAE
optimizer = torch.optim.Adam(pred_model.parameters(), lr=0.001)

losses = []
epochs = 50000
patience = 100  # Number of epochs to wait for improvement before stopping
best_loss = float('inf')
stop_counter = 0

for epoch in range(epochs):
    optimizer.zero_grad()
    # import ipdb; ipdb.set_trace()
    # Forward pass
    y_pred = pred_model(x_train.float().to(device))
    # Compute the loss
    loss = criterion(y_pred.reshape(-1), y_train.float().to(device))
    # Backward pass
    loss.backward()
    # Update the parameters
    optimizer.step()
    # Append the loss to the list
    losses.append(loss.item())
    if (epoch + 1) % 250 == 0:
        print(f"Epoch {epoch + 1}, MSE Loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1}, MAE Loss: {nn.L1Loss()(y_pred.reshape(-1), y_train.float().to(device)).item():.4f}")
    
    
    # Check if the validation loss has improved
    if loss.item() < best_loss:
        best_loss = loss.item()
        stop_counter = 0
    else:
        stop_counter += 1
    if stop_counter >= patience:
        print("Early stopping triggered")
        break

with torch.no_grad():
    model.eval()
    y_pred_test = pred_model(x_test.float().to(device))

# Print test loss
print(f'Test MSE loss: {criterion(y_pred_test.reshape(-1), y_test.float().to(device)).item()}')
print(f'Test MAE loss: {nn.L1Loss()(y_pred_test.reshape(-1), y_test.float().to(device)).item()}')

fig, ax = plt.subplots(1, 2, figsize=(10, 3))

m = Basemap(projection='cyl', resolution='c', ax=ax[0])
m.drawcoastlines()
ax[0].scatter(coords_test[:,1], coords_test[:,0], c=y_test, s=5)
ax[0].set_title('True')

m = Basemap(projection='cyl', resolution='c', ax=ax[1])
m.drawcoastlines()
ax[1].scatter(coords_test[:,1].cpu().numpy(), coords_test[:,0].cpu().numpy(), c=y_pred_test.reshape(-1).cpu().numpy(), s=5)
ax[1].set_title('Wind')
plt.savefig('../AirTemperature_predicted_day_csp.jpg',dpi=1000)
