from monai.networks.nets import VNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preprocess import prepare
from utilities import train


data_dir = 'C:/Users/Asus/Desktop/hippocampus segmentation/data/datasets'
model_dir = 'C:/Users/Asus/Desktop/hippocampus segmentation/data/results'
data_in = prepare(data_dir, cache=True)

device = torch.device("cpu")
model = VNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    act=('relu', {'inplace': True}),
    dropout_prob=0.5,
    dropout_dim=3,
)


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 100, model_dir)