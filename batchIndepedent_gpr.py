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



rig_fileName="cube_rig_data_05.csv"
jnt_fileName="cube_jnt_data_05.csv"

#def gpr(rig_fileName="cube_rig_data_05.csv", jnt_fileName="cube_jnt_data_05.csv", anim_path=None, plots=False):
rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", rig_fileName)
jnt_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", jnt_fileName)

rig_dataset = pd.read_csv(rig_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])
jnt_dataset = pd.read_csv(jnt_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

# get number of objs in first column and mult it with len of data entries
train_x_dimension =  len(np.unique(jnt_dataset.iloc[:, :1])) * len(jnt_dataset.iloc[:, 2:].values[0])
train_x = torch.from_numpy(np.array(jnt_dataset.iloc[:, 2:]).reshape(-1, train_x_dimension)).float()
train_x = train_x.to(device)

x_means = train_x.mean(1, keepdim=True)
x_deviations = train_x.std(1, keepdim=True)
train_x_norm = (train_x - x_means) / x_deviations
train_x_norm = torch.nn.functional.normalize(train_x)
train_x_norm = (train_x - train_x.min()) / (train_x.max() - train_x.min())


# get number of objs in first column and mult it with dimension of data
train_y_dimension = rig_dataset.filter(items=["dimension"]).values[0][0] * len(np.unique(rig_dataset.iloc[:, :1]))
rig_train_columns = [col for col in rig_dataset.columns.values if "_value" in col]
train_y = torch.from_numpy(np.array(rig_dataset.filter(items=rig_train_columns)).reshape(-1, train_y_dimension)).float()
train_y = train_y.to(device)

#y_means = train_y.mean(1, keepdim=True)
#y_deviations = train_y.std(1, keepdim=True)
#train_y_norm = (train_y - y_means) / y_deviations


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y_dimension]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([train_y_dimension])),
            batch_shape=torch.Size([train_y_dimension])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension)
likelihood = likelihood.to(device)
model = BatchIndependentMultitaskGPModel(train_x=train_y, train_y=train_x_norm, likelihood=likelihood)
model = model.to(device)


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # lr = learning rate

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 250
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()


plots = True
if plots:
    # Initialize plots
    f, plot_list = plt.subplots(train_x_dimension, 1, figsize=(7.5, 5*train_x_dimension))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x_tensor = [torch.linspace(-50, 50, 100) for i in range(train_x_dimension)]
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
    for i in range(train_y_dimension):
        ax_plot(plot_list[i], i, train_y[:, i], train_x[:, i], predictions, 'Observed Values (Likelihood)', -50, 50)


    #return likelihood, model



#if __name__ == "__main__":
#    gpr(rig_data=pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", file_name),
#        jnt_data=pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", file_name), plots=True)