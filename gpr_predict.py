import math
import torch
import gpytorch
import numpy as np
import pandas as pd
import csv
from importlib import reload
import pathlib
import os
import torch.autograd.profiler as profiler


import gpr_model as gpr
reload(gpr)


# enable GPU/CUDA if available
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu"
dev = "cpu" 
device = torch.device(dev)


def matrix_to_6d(rot_mat):
    # Use the first two columns of the rotation matrix to get the 6D representation
    return rot_mat[:, :2].reshape(-1)


def _6d_to_matrix(rot_6d):
    # Reshape the 6D representation back to a 3x2 matrix
    mat = rot_6d.view(3, 2)

    # Calculate the third column of the rotation matrix as the cross product of the first two columns
    third_col = torch.cross(mat[:, 0], mat[:, 1]).unsqueeze(1)

    # Construct the full rotation matrix
    return torch.cat((mat, third_col), dim=1)


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

#anim_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "irm_anim_data.csv")
def predict_data(anim_path):
    # load trained model
    rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", "irm_rig_data.csv")
    jnt_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", "irm_jnt_data.csv")
    train_x = gpr.build_train_x_tensor(jnt_file)
    train_y, train_y_dimension = gpr.build_train_y_tensor(rig_file)

    model_file = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "trained_model.pt")
    state_dict = torch.load(model_file)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension)
    likelihood.to(device)

    #model = gpr.MultitaskGPModel(train_x, train_y, likelihood, train_y_dimension)
    model = gpr.BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, train_y_dimension)
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    likelihood.eval()


    # load anim data
    anim_dataset = pd.read_csv(anim_path, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    # get number of objs in first column and mult it with len of data entries
    anim_x_depth =  len(np.unique(anim_dataset.iloc[:, :1]))
    anim_data_frames = int(len(anim_dataset)/anim_x_depth)
    anim_dimension = len(anim_dataset.iloc[:, 3:].values[0])

    raw_anim_x = np.array(anim_dataset.iloc[:, 3:]).reshape(-1, anim_dimension)

    # normalisation: -1.0 to 1.0
    attr_list = anim_dataset.columns.values[3:].tolist()
    normalise_index_list = [attr_list.index(attr) for attr in attr_list if not "rotMtx_" in attr]

    #anim_x_min, anim_x_max = -170.0, 35.0
    #anim_x_min, anim_x_max = -50.0, 200.0
    anim_x_min, anim_x_max = -50.0, 50.0
    new_min, new_max = -1.0, 1.0

    normalised_anim_x = raw_anim_x
    for entry_index, entry in enumerate(normalised_anim_x):
        for value_index, value in enumerate(entry):
            if value_index in normalise_index_list:
                value_norm = (value - anim_x_min) / (anim_x_max - anim_x_min) * (new_max - new_min) + new_min
                normalised_anim_x[entry_index][value_index] = value_norm

    cleaned_anim_x = np.array([entry for row in normalised_anim_x for entry in row if str(entry) != "nan"]) # remove n/a entries from data
    anim_x_rotMtx = torch.from_numpy(cleaned_anim_x.reshape(anim_data_frames, -1)).float()

    anim_x_norm = convert_tensor_to_6d(anim_x_rotMtx, anim_x_depth).reshape(anim_data_frames, -1)
    anim_x_norm = anim_x_norm.to(device)


    # get predict values from trained model
    #anim_x_norm = torch.randn(2, 210)
    predict_mean = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #for tensor in anim_x_norm:
        #    print("TENSOR")
        #    predict_y = likelihood(model(tensor.reshape(1, -1)))
        #    predict_mean.append(predict_y.mean.reshape(-1).tolist())
        predict_y = likelihood(model(anim_x_norm))
        predict_mean = predict_y.mean.tolist()

    print("DONE")
    # get rig dataset used in training for building predict data
    rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", "irm_rig_data.csv")
    train_rig_df = pd.read_csv(rig_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    rig_obj_names = np.unique(train_rig_df.iloc[:, 1]) # get names of rig objects for data mapping later
    rig_dimension_list = train_rig_df.filter(items=["dimension"]).values[:len(rig_obj_names)] # get dimension per rig object for data mapping


    # convert rot 6d back to rot matrix
    predict_6d = torch.tensor(np.array(predict_mean)).reshape(anim_data_frames, anim_x_depth, -1)
    predict_rotMtx = []
    for entry in predict_6d:
        temp = []
        for obj in entry:
            rot_mtx = _6d_to_matrix(obj)
            temp.append(rot_mtx.cpu().numpy())
        predict_rotMtx.append(temp)
    predict_mean = torch.from_numpy(np.array(predict_rotMtx)).reshape(anim_data_frames, -1).float().tolist()




    predict_data = []
    for data in predict_mean:
        start_range = 0
        # reshape data per frame into dimensions of each rig object as in train rig data
        for i, dimension in enumerate(rig_dimension_list):
            # isolate predict values of single obj
            dimension = int(dimension)
            predict_values = data[start_range:(start_range + dimension)]
            start_range += dimension

            # use train data structure as base and replace values with predicted values
            # n/a will stay in missing columns
            # No., rigName and dimension will be the same aswell
            frame_data = train_rig_df.iloc[i].values.tolist()
            value_index = 0
            for value in frame_data[3:]:
                if not np.isnan(value):
                    frame_data[frame_data.index(value)] = predict_values[value_index] 
                    value_index += 1
            predict_data.append(frame_data)


    # save anim data for Maya
    predict_name = "irm_predict_data.csv"
    predict_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "predict_data", predict_name)
    predict_header = train_rig_df.columns.values # use header of rig train data for predict header


    with open(predict_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(predict_header)
        writer.writerows(predict_data)



if __name__=="__main__":
    predict_data(anim_path=pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "irm_anim_data.csv"))