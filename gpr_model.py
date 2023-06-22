"""
-----------------------------------------------------------------------------
This file has been developed within the scope of the
Technical Director course at Filmakademie Baden-Wuerttemberg.
http://technicaldirector.de

Written by Lukas Kapp
Copyright (c) 2023 Animationsinstitut of Filmakademie Baden-Wuerttemberg
-----------------------------------------------------------------------------
"""

import torch
import gpytorch
import numpy as np
import pandas as pd
import pathlib
import os

import utils.pytorch as torchUtils

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, force_cpu, train_x_dimension, train_y_dimension,
                x_min,x_max, x_mean, y_min, y_max, y_mean):
        super(BatchIndependentMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y_dimension]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([train_y_dimension])),
            batch_shape=torch.Size([train_y_dimension])
        )

        self.register_buffer('force_cpu', torch.tensor([force_cpu]))

        self.register_buffer('train_x', train_x)
        self.register_buffer('x_min', x_min)
        self.register_buffer('x_max', x_max)
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_dim', torch.tensor([train_x_dimension]))
        
        self.register_buffer('train_y', train_y)
        self.register_buffer('y_min', y_min)
        self.register_buffer('y_max', y_max)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_dim', torch.tensor([train_y_dimension]))


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


def build_data_tensor(dataset_path, min_val=None, max_val=None, mean_val=None):
    ### BUILD DATA TENSOR ###
    dataset = pd.read_csv(dataset_path, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])
    
    # number of objs =  length of unique items of first column
    numObjs = len(np.unique(dataset.iloc[:, 0]))

    # get dimensions of each obj and get sum of it
    dimension = sum(dataset.filter(items=["dimension"]).values[:numObjs])[0]
    highest_dim = max(dataset.filter(items=["dimension"]).values[:numObjs])[0]
    
    # create list with entries of all attribute columns
    raw_tensor = np.array(dataset.iloc[:, 3:].values).reshape(-1, numObjs*highest_dim)
    attr_list = dataset.columns.values.tolist()
    rotMtx_index_list = [attr_list.index(attr) for attr in attr_list if "rotMtx_" in attr]

    # extract values before and after rotMtx
    before_rotMtx_tensor = torch.from_numpy(dataset.iloc[:, 3:rotMtx_index_list[0]].values)
    after_rotMtx_tensor = torch.from_numpy(dataset.iloc[:, rotMtx_index_list[-1]+1:].values)

    # extract rot matrix and convert to quaternion
    rotMtx_tensor = torch.from_numpy(dataset.iloc[:, rotMtx_index_list[0]:rotMtx_index_list[-1]+1].values).reshape(-1, 3, 3)
    quat_tensor = torchUtils.batch_rotation_matrix_to_quaternion(rotMtx_tensor)
    #quat_tensor = torchUtils.matrix_to_6d(rotMtx_tensor).reshape(-1, 6)

    # concatenate tensors back; reduce dimension since quat is only 4 entries compared to 9 of rot mtx
    concat_tensor = torch.cat((before_rotMtx_tensor, quat_tensor, after_rotMtx_tensor), dim=1)
    quat_dim = dimension - (5 * numObjs)
    #quat_dim = dimension - (3 * numObjs)

    # remove n/a entries from data
    cleaned_tensor = torch.from_numpy(np.array([entry for row in concat_tensor.tolist() for entry in row if str(entry) != "nan"]).reshape(-1, quat_dim))

    # normalize tensor
    if min_val is None or max_val is None or mean_val is None:
        min_val, max_val, mean_val = torchUtils.calculate_min_max_mean(cleaned_tensor)
    norm_tensor = torchUtils.normalize_tensor(cleaned_tensor, min_val, max_val, mean_val)

    return norm_tensor.float(), quat_dim, min_val.float(), max_val.float(), mean_val.float(), concat_tensor.float()



#rig_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig/irm_rig_data.csv").as_posix()
#jnt_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt/irm_jnt_data.csv").as_posix()
#model_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "trained_model/trained_model.pt").as_posix()
#lr = 0.1
#epochs = 100
#force_cpu = False

def train_model(rig_path, jnt_path, model_path, lr, epochs, force_cpu):
    ## enable GPU/CUDA if available
    if torch.cuda.is_available() and not force_cpu: 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)

    # build tensors
    train_x, train_x_dimension, x_min, x_max, x_mean, x_concat = build_data_tensor(jnt_path)
    train_x = train_x.to(device)

    train_y, train_y_dimension, y_min, y_max, y_mean, y_concat = build_data_tensor(rig_path)
    train_y = train_y.to(device)


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension).to(device)
    model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, force_cpu, train_x_dimension,
                                            train_y_dimension, x_min,x_max, x_mean, y_min, y_max, y_mean).to(device)


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # lr = learning rate

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(epochs):       
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, epochs, loss.item()))
        optimizer.step()
        print(f"PROGRESS {100.0 * (i + 1) / epochs}")   


    # Set into eval mode
    model.eval()
    likelihood.eval()

    torch.save(model.state_dict(), model_path) # save trained model parameters
