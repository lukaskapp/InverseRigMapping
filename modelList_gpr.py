import math
import torch
import gpytorch
import numpy as np
import pandas as pd
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
device = torch.device(dev)

#rig_fileName="cube_rig_data_05.csv"
#jnt_fileName="cube_jnt_data_05.csv"
#plots=True

def gpr(rig_fileName="cube_rig_data_05.csv", jnt_fileName="cube_jnt_data_05.csv", anim_path=None, plots=False):
    rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", rig_fileName)
    jnt_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", jnt_fileName)

    rig_dataset = pd.read_csv(rig_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])
    jnt_dataset = pd.read_csv(jnt_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    # get number of objs in first column and mult it with len of data entries
    train_x_dimension =  len(np.unique(jnt_dataset.iloc[:, :1])) * len(jnt_dataset.iloc[:, 2:].values[0])
    train_x = torch.from_numpy(np.array(jnt_dataset.iloc[:, 2:]).reshape(-1, train_x_dimension)).float()
    train_x = train_x.to(device)
    train_x_trans = train_x[:, :3]
    train_x_rot = train_x[:, 3:]

    x_trans_means = train_x_trans.mean(1, keepdim=True)
    x_trans_deviations = train_x_trans.std(1, keepdim=True)
    #train_x_trans_norm = (train_x_trans - x_trans_means) / x_trans_deviations
    #train_x_trans_norm = torch.nn.functional.normalize(train_x_trans)
    train_x_trans_norm = (train_x_trans - train_x_trans.min()) / (train_x_trans.max() - train_x_trans.min())
    train_x_rot_norm = (train_x_rot - train_x_rot.min()) / (train_x_rot.max() - train_x_rot.min())

    train_x_norm = torch.cat((train_x_trans_norm, train_x_rot_norm), 1)

    # get number of objs in first column and mult it with dimension of data
    train_y_dimension = rig_dataset.filter(items=["dimension"]).values[0][0] * len(np.unique(rig_dataset.iloc[:, :1]))
    rig_train_columns = [col for col in rig_dataset.columns.values if "_value" in col]
    train_y = torch.from_numpy(np.array(rig_dataset.filter(items=rig_train_columns)).reshape(-1, train_y_dimension)).float()
    train_y = train_y.to(device)

    #y_means = train_y.mean(1, keepdim=True)
    #y_deviations = train_y.std(1, keepdim=True)
    #train_y_norm = (train_y - y_means) / y_deviations



    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood_list = torch.nn.ParameterList()
    model_list = torch.nn.ModuleList()
    for i in range(train_y_dimension):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood = likelihood.to(device)
        model = ExactGPModel(train_x[:, i], train_y[:, i], likelihood)
        model = model.to(device)

        likelihood_list.append(likelihood)
        model_list.append(model)



    model = gpytorch.models.IndependentModelList(*model_list)
    model = model.to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)
    likelihood = likelihood.to(device)


    from gpytorch.mlls import SumMarginalLogLikelihood

    mll = SumMarginalLogLikelihood(likelihood, model)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # lr = learning rate

    training_iterations = 250
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()



    # Set into eval mode
    model.eval()
    likelihood.eval()


    #plots = True
    if plots:
        num_of_plots = 9
        # Initialize plots
        f, axs = plt.subplots(num_of_plots, 1, figsize=(7.5, 5*num_of_plots))

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(train_x.min().cpu(), train_x.max().cpu(), 100)
            test_x = test_x.to(device)
            test_x_tensor = [test_x for i in range(train_y_dimension)]
            predictions = likelihood(*model(*test_x_tensor))


        # Define plotting function
        def ax_plot(ax, train_y, train_x, prediction, title, min_y, max_y):
            # Get lower and upper confidence bounds
            lower, upper = prediction.confidence_region()
            mean = prediction.mean
            # Plot training data as black stars
            ax.plot(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy(), 'k*')
            # Predictive mean as blue line
            ax.plot(test_x.detach().cpu().numpy(), mean.detach().cpu().numpy(), 'b')
            # Shade in confidence
            ax.fill_between(test_x.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy(), alpha=0.5)
            ax.set_ylim([min_y, max_y])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            ax.set_title(title)

        # Plot both tasks
        for submodel, prediction, ax in zip(model.models[:num_of_plots], predictions[:num_of_plots], axs[:num_of_plots]):
            ax_plot(ax, submodel.train_targets, submodel.train_inputs[0], prediction, 'Observed Values (Likelihood)', prediction.confidence_region()[0].min().cpu()*1.1,prediction.confidence_region()[1].max().cpu()*1.1)


    return likelihood, model



if __name__ == "__main__":
    gpr(rig_fileName="cube_rig_data_05.csv", jnt_fileName="cube_jnt_data_05.csv", anim_path=None, plots=True)