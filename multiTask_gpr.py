import math
import torch
import gpytorch
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
#import multiquadric_kernel as mqk
from importlib import reload
import pathlib
import os

#reload(mqk)


# enable GPU/CUDA if available
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
dev = "cpu"
device = torch.device(dev)


#rig_fileName="irm_rig_data.csv"
#jnt_fileName="irm_jnt_data.csv"
#plots=True

def gpr(rig_fileName="irm_rig_data.csv", jnt_fileName="irm_jnt_data.csv", anim_path=None, plots=False):
    rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", rig_fileName)
    jnt_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", jnt_fileName)

    rig_dataset = pd.read_csv(rig_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])
    jnt_dataset = pd.read_csv(jnt_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    # get number of objs in first column and mult it with len of data entries
    # tensor = all values of all jnts on one frame/entry
    train_x_dimension =  len(np.unique(jnt_dataset.iloc[:, :1])) * len(jnt_dataset.iloc[:, 2:].values[0])
    train_x = torch.from_numpy(np.array(jnt_dataset.iloc[:, 2:]).reshape(-1, train_x_dimension)).float()
    train_x = train_x.to(device)
    train_x_trans = train_x[:, :3]
    train_x_rot = train_x[:, 3:]

    #x_trans_means = train_x_trans.mean(1, keepdim=True)
    #x_trans_deviations = train_x_trans.std(1, keepdim=True)
    #train_x_trans_norm = (train_x_trans - x_trans_means) / x_trans_deviations
    #train_x_trans_norm = torch.nn.functional.normalize(train_x_trans)
    #train_x_trans_norm = (train_x_trans - train_x_trans.min()) / (train_x_trans.max() - train_x_trans.min())

    #x_rot_means = train_x_rot.mean(0, keepdim=True)
    #x_rot_deviations = train_x_rot.std(0, keepdim=True)
    #train_x_rot_norm = (train_x_rot - x_rot_means) / x_rot_deviations
    #train_x_rot_norm = torch.nn.functional.normalize(train_x_rot)
    #train_x_rot_norm = (train_x_rot - train_x_rot.min()) / (train_x_rot.max() - train_x_rot.min())

    # normalize inputs to range -1.0 1.0
    new_min, new_max = -1.0, 1.0

    x_trans_min, x_trans_max = train_x_trans.min(), train_x_trans.max()
    train_x_trans_norm = (train_x_trans - x_trans_min) / (x_trans_max - x_trans_min) * (new_max - new_min) + new_min
        
    #x_rot_min, x_rot_max = train_x_rot.min(), train_x_rot.max()
    #train_x_rot_norm = (train_x_rot - x_rot_min) / (x_rot_max - x_rot_min) * (new_max - new_min) + new_min
        
    train_x_norm = torch.cat((train_x_trans_norm, train_x_rot), 1)


    #train_x_norm = train_x_rot
    #train_x_dimension = 9

    # get number of objs in first column and mult it with dimension of data
    train_y_dimension = rig_dataset.filter(items=["dimension"]).values[0][0] * len(np.unique(rig_dataset.iloc[:, :1]))
    rig_train_columns = [col for col in rig_dataset.columns.values if "_value" in col]
    train_y = torch.from_numpy(np.array(rig_dataset.filter(items=rig_train_columns)).reshape(-1, train_y_dimension)).float()
    #train_y = train_y[:, 3:]
    #train_y_dimension = 9
    train_y = train_y.to(device)


    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=train_y_dimension
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=train_y_dimension, rank=1
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension)
    likelihood.to(device)
    model = MultitaskGPModel(train_x_norm, train_y, likelihood)
    model.to(device)


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)  # lr = learning rate

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iterations = 500
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x_norm)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()


    # Set into eval mode
    model.eval()
    likelihood.eval()

    if plots:
        num_of_plots = 3
        # Initialize plots
        f, axs = plt.subplots(num_of_plots, 1, figsize=(7.5, 5*num_of_plots))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x_tensor = [torch.linspace(train_x_rot.min().cpu(), train_x_rot.max().cpu(), 100) for i in range(train_x_dimension)]
            test_x = torch.stack(test_x_tensor, -1).float()
            test_x = test_x.to(device)
            predictions = likelihood(model(test_x))

        # Define plotting function
        def ax_plot(ax, index, train_y, train_x, prediction, title, min_y, max_y):
            # Get lower and upper confidence bounds
            lower, upper = prediction.confidence_region()
            mean = prediction.mean
            # Plot training data as black stars
            ax.plot(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy(), 'k*')
            # Predictive mean as blue line
            ax.plot(test_x[:, index].detach().cpu().numpy(), mean[:, index].detach().cpu().numpy(), 'b')
            # Shade in confidence
            ax.fill_between(test_x[:, index].detach().cpu().numpy(), lower[:, index].detach().cpu().numpy(), upper[:, index].detach().cpu().numpy(), alpha=0.5)
            ax.set_ylim([min_y, max_y])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            ax.set_title(title)

        # Plot both tasks
        for i in range(num_of_plots):
            ax_plot(axs[i], i, train_y[:, i], train_x_rot[:, i], predictions, 'Observed Values (Likelihood)', -50, 50)
        #ax_plot(axs[1], 1, train_y[:, 1], train_x[:, 0], predictions, 'Observed Values (Likelihood)', -40, 40)
        
    
    return likelihood, model, x_trans_min, x_trans_max



#if __name__ == "__main__":
#    gpr(rig_fileName="irm_rig_data.csv", jnt_fileName="irm_jnt_data.csv", anim_path=None, plots=True)
