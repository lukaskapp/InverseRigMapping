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


# enable GPU/CUDA if available
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
dev = "cpu" 
device = torch.device(dev)


class MultiquadricKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    
    def __init__(
        self,
        batch_shape=torch.Size(),
        lengthscale_prior=None,
        lengthscale_constraint=None,
        **kwargs
    ):
        super().__init__(
            has_lengthscale=self.has_lengthscale,
            batch_shape=batch_shape,
            **kwargs
        )
        
        if lengthscale_constraint is None:
            lengthscale_constraint = gpytorch.constraints.Positive()
            
        self.raw_lengthscale = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self.lengthscale_constraint = lengthscale_constraint
        self.register_parameter(
            name="raw_lengthscale",
            parameter=self.raw_lengthscale
        )
        
        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda: self.lengthscale,
                lambda v: self._set_lengthscale(v),
            )
            
        self.lengthscale = self.lengthscale_constraint.transform(self.raw_lengthscale)
        
    def forward(self, x1, x2, diag=False, **params):
        lengthscale = self.lengthscale.unsqueeze(-1)
        x1_ = x1.div(lengthscale)
        x2_ = x2.div(lengthscale)
        if diag:
            return ((x1_ - x2_) ** 2).sum(dim=-1).sqrt()
        else:
            return ((x1_.unsqueeze(-2) - x2_.unsqueeze(-3)) ** 2).sum(dim=-1).sqrt()



def farthest_point_sampling(points, k):
    """Farthest Point Sampling (FPS) algorithm.
    
    Args:
        points (np.ndarray or torch.Tensor): The dataset to subsample from, with shape (N, D).
        k (int): The number of points to select.
        
    Returns:
        np.ndarray or torch.Tensor: The selected points, with shape (k, D).
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    N, D = points.shape
    selected_indices = [np.random.randint(N)]  # Start with a random point
    distances = np.full((N,), np.inf)

    for _ in range(k - 1):
        # Compute distances to the last selected point
        new_distances = np.linalg.norm(points - points[selected_indices[-1]], axis=1)
        
        # Update the minimum distances
        np.minimum(distances, new_distances, out=distances)
        
        # Select the point with the largest minimum distance
        selected_indices.append(np.argmax(distances))
    
    return points[selected_indices]


class IrmDataLoader(torch.utils.data.Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        x = self.input_data[index]
        y = self.target_data[index]
        return x, y

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, train_y_dimension):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y_dimension]))
        #self.covar_module = gpytorch.kernels.ScaleKernel(
        #    MultiquadricKernel(batch_shape=torch.Size([train_y_dimension])),
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



rig_fileName="irm_rig_data.csv"
jnt_fileName="irm_jnt_data.csv"
model_file="trained_model.pt"
#def train_model(rig_fileName="irm_rig_data.csv", jnt_fileName="irm_jnt_data.csv",  model_file="trained_model.pt"):
rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", rig_fileName)
jnt_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", jnt_fileName)

train_x = build_train_x_tensor(jnt_file)
#train_x = train_x.float()
#train_x = train_x.cpu().numpy()
train_y, train_y_dimension = build_train_y_tensor(rig_file)
#train_y = train_y.float()
#train_y = train_y.cpu().numpy()

#k = 10
#train_x_subsampled_indices = farthest_point_sampling(train_x, k)
#train_x_subsampled = train_x.index_select(0, torch.tensor(train_x_subsampled_indices))
#train_y_subsampled = train_y.index_select(0, torch.tensor(train_x_subsampled_indices))


#dataset = IrmDataLoader(train_x, train_y)
#train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

#dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension)
likelihood.to(device)
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
    #    # Update the model with the current batch of data
    #    model.set_train_data(inputs=batch_train_x, targets=batch_train_y, strict=False)

        # Perform your training steps here
    #    optimizer.zero_grad()
    #    with gpytorch.settings.use_toeplitz(False), torch.autograd.set_detect_anomaly(True):
    #        output = model(batch_train_x)
    #        loss = -mll(output, batch_train_y)
    #    loss.backward()
    #    optimizer.step()
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




