import math
import torch
import gpytorch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from importlib import reload
import pathlib
import os
from torch.cuda.amp import autocast, GradScaler

import utils.multiquadric_kernel as mqk
reload(mqk)

import utils.dataLoader as dataLoader
reload(dataLoader)

import utils.fps as fps
reload(fps)

# enable GPU/CUDA if available
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
dev = "cpu" 
device = torch.device(dev)



class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, train_y_dimension):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y_dimension]))
        #self.covar_module = gpytorch.kernels.ScaleKernel(
        #    mqk.MultiquadricKernel(batch_shape=torch.Size([train_y_dimension])),
        #    batch_shape=torch.Size([train_y_dimension])
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


def matrix_to_6d(rot_mat):
    # Use the first two columns of the rotation matrix to get the 6D representation
    return rot_mat[:, :2].reshape(-1)


def convert_tensor_to_6d(tensor, numObjs):
    tensor_rotMtx = tensor.reshape(-1, numObjs, 3, 3)
    tensor_6d = []
    for entry in tensor_rotMtx:
        temp = []
        for obj in entry:
            rot_6d = matrix_to_6d(obj)
            temp.append(rot_6d.cpu().numpy())
        tensor_6d.append(temp)
    out_tensor = torch.from_numpy(np.array(tensor_6d)).float()

    return out_tensor


def build_train_x_tensor(jnt_file):
    ### BUILD JOINT DATA TENSOR ###
    jnt_dataset = pd.read_csv(jnt_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    train_x_numObjs = len(np.unique(jnt_dataset.iloc[:, 0])) # number of objs =  length of unique items of first column
    train_x_dimension = sum(jnt_dataset.filter(items=["dimension"]).values[:train_x_numObjs])[0] # get dimensions of each obj and get sum of it
    
    raw_train_x = jnt_dataset.iloc[:, 3:].values.tolist() # create list with entries of all attribute columns
    
    
    # normalisation: -1.0 to 1.0
    attr_list = jnt_dataset.columns.values[3:].tolist()
    normalise_index_list = [attr_list.index(attr) for attr in attr_list if not "rotMtx_" in attr]
    
    #train_x_min, train_x_max = -170.0, 35.0
    #train_x_min, train_x_max = -50.0, 200.0
    train_x_min, train_x_max = -50.0, 50.0
    #train_x_min, train_x_max = train_x_trans.min(), train_x_trans.max()
    new_min, new_max = -1.0, 1.0

    normalised_train_x = raw_train_x
    for entry_index, entry in enumerate(normalised_train_x):
        for value_index, value in enumerate(entry):
            if value_index in normalise_index_list:
                value_norm = (value - train_x_min) / (train_x_max - train_x_min) * (new_max - new_min) + new_min
                normalised_train_x[entry_index][value_index] = value_norm
    
    cleaned_train_x = np.array([entry for row in normalised_train_x for entry in row if str(entry) != "nan"]) # remove n/a entries from data
    train_x_rotMtx = torch.from_numpy(cleaned_train_x.reshape(-1, train_x_dimension)).float()
    
    train_x = convert_tensor_to_6d(train_x_rotMtx, train_x_numObjs).reshape(-1, train_x_dimension-train_x_numObjs*3)
    train_x = train_x.to(device)
   
    return train_x


def build_train_y_tensor(rig_file):
    ### BUILD RIG DATA TENSOR ###
    rig_dataset = pd.read_csv(rig_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])
   
    train_y_numObjs = len(np.unique(rig_dataset.iloc[:, 0])) # number of objs =  length of unique items of first column
    train_y_dimension = sum(rig_dataset.filter(items=["dimension"]).values[:train_y_numObjs])[0] # get dimensions of each control and get sum of it

    raw_train_y = rig_dataset.iloc[:, 3:].values.tolist() # create list with entries of all attribute columns
    cleaned_train_y = np.array([entry for row in raw_train_y for entry in row if str(entry) != "nan"]) # remove n/a entries from data
    train_y_rotMtx = torch.from_numpy(cleaned_train_y.reshape(-1, train_y_dimension)).float() # reshape data to fit train_y_dimension
    
    train_y = convert_tensor_to_6d(train_y_rotMtx, train_y_numObjs).reshape(-1, train_y_dimension-train_y_numObjs*3)
    train_y = train_y.to(device)

    return train_y, train_y_dimension-train_y_numObjs*3



def plot_data():
    num_of_plots = 3
    # Initialize plots
    f, axs = plt.subplots(num_of_plots, 1, figsize=(7.5, 5*num_of_plots))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x_tensor = [torch.linspace(train_x_rot.min().cpu(), train_x_rot.max().cpu(), 100) for i in range(train_x_dimension)]
        test_x = torch.stack(test_x_tensor, -1).half()
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


#rig_fileName="irm_rig_data.csv"
#jnt_fileName="irm_jnt_data.csv"
#model_file="trained_model.pt"
def train_model(rig_fileName="irm_rig_data.csv", jnt_fileName="irm_jnt_data.csv",  model_file="trained_model.pt"):
    rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", rig_fileName)
    jnt_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", jnt_fileName)

    train_x = build_train_x_tensor(jnt_file)
    #train_x = train_x.float()
    #train_x = train_x.cpu().numpy()
    train_y, train_y_dimension = build_train_y_tensor(rig_file)
    #train_y = train_y.float()
    #train_y = train_y.cpu().numpy()

    #k = 10
    #train_x_subsampled_indices = fps.farthest_point_sampling(train_x, k)
    #train_x_subsampled = train_x.index_select(0, torch.tensor(train_x_subsampled_indices))
    #train_y_subsampled = train_y.index_select(0, torch.tensor(train_x_subsampled_indices))



    #dataset = dataLoader.IrmDataLoader(train_x, train_y)
    #train_dataset = torch.utils.data.TensorDataset(train_x_subsampled, train_y_subsampled)

    #dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension)
    likelihood.to(device)
    #model = MultitaskGPModel(train_x, train_y, likelihood, train_y_dimension)
    model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, train_y_dimension)
    #model = BatchIndependentMultitaskGPModel(None, None, likelihood, train_y_dimension)
    model.to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # lr = learning rate

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    scaler = GradScaler()
    training_iterations = 50
    for i in range(training_iterations):
        #for batch_train_x, batch_train_y in dataloader:
            # Update the model with the current batch of data
            #model.set_train_data(inputs=batch_train_x, targets=batch_train_y, strict=False)

            # Perform your training steps here
            #optimizer.zero_grad()
            #with gpytorch.settings.use_toeplitz(False), torch.autograd.set_detect_anomaly(True):
            #    output = model(batch_train_x)
            #    loss = -mll(output, batch_train_y)
            #loss.backward()
            #optimizer.step()
        #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))

        
        optimizer.zero_grad()

        #with autocast():
        #    output = model(train_x)
        #    loss = -mll(output, train_y)
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()


        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    save_path = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), model_file)
    torch.save(model.state_dict(), save_path) # save trained model parameters



if __name__ == "__main__":
    train_model(rig_fileName="irm_rig_data.csv", jnt_fileName="irm_jnt_data.csv", model_file="trained_model.pt")
