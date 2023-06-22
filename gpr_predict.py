import math
import torch
import gpytorch
import numpy as np
import pandas as pd
import csv
from importlib import reload
import pathlib
import os

import gpr_model as gpr
reload(gpr)

import utils.pytorch as torchUtils
reload(torchUtils)


#anim_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "irm_anim_data.csv").as_posix()
#model_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "trained_model/trained_model.pt").as_posix()
#rig_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig/irm_rig_data.csv").as_posix()
def predict_data(anim_path, model_path, rig_path):
    # load trained model
    state_dict = torch.load(model_path)
    force_cpu = state_dict["force_cpu"]

    # enable GPU/CUDA if available
    if torch.cuda.is_available() and not force_cpu: 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)

    train_x = state_dict["train_x"]
    train_x = train_x.to(device)
    train_y = state_dict["train_y"]
    train_y = train_y.to(device)
    train_x_dimension = state_dict["x_dim"]
    train_y_dimension = state_dict["y_dim"]

    x_min = state_dict["x_min"].to(device)
    x_max = state_dict["x_max"].to(device)
    x_mean = state_dict["x_mean"].to(device)

    y_min = state_dict["y_min"].to(device)
    y_max = state_dict["y_max"].to(device)
    y_mean = state_dict["y_mean"].to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension).to(device)

    model = gpr.BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, force_cpu, train_x_dimension,
                                            train_y_dimension, x_min,x_max, x_mean, y_min, y_max, y_mean).float().to(device)
    model.load_state_dict(state_dict)

    model.eval()
    likelihood.eval()
    print("PROGRESS 20")

    # load anim data
    print("PROGRESS 40")

    anim_dataset = pd.read_csv(anim_path, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])
    print("PROGRESS 40")

    anim_numObjs = len(np.unique(anim_dataset.iloc[:, 1]))
    print("PROGRESS 40")

    anim_frame_len = int(len(anim_dataset.iloc[:, 1])/anim_numObjs)
    print("PROGRESS 40")

    anim_x, anim_quat_dim, min_val, max_val, mean_val, anim_concat = gpr.build_data_tensor(anim_path, min_val=x_min.to("cpu"), max_val=x_max.to("cpu"), mean_val=x_mean.to("cpu"))
    anim_x = anim_x.to(device)

    # get predict values from trained model
    print("PROGRESS 40")
    predict_mean = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predict_y = likelihood(model(anim_x))
        predict_mean = predict_y.mean.tolist()
    print("PROGRESS 70")


    # get rig dataset used in training for building predict data
    train_rig_df = pd.read_csv(rig_path, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])
    rig_x, rig_quat_dim, rig_min, rig_max, rig_mean, rig_concat = gpr.build_data_tensor(rig_path)

    rig_numObjs = len(np.unique(train_rig_df.iloc[:, 1]))
    rig_highest_dim = max(train_rig_df.filter(items=["dimension"]).values[:rig_numObjs])[0]

    # denormalise tensor
    predict_tensor = torch.from_numpy(np.array(predict_mean)).to(device)
    denorm_predict = torchUtils.denormalize_tensor(predict_tensor, y_min, y_max, y_mean)

    # convert predict tensor to same structure of dataframe (with nan values)
    attr_list = train_rig_df.columns.values[3:].tolist()
    rotMtx_start = [attr_list.index(attr) for attr in attr_list if "rotMtx_" in attr][0]

    rig_nan_tensor = torch.tensor(rig_concat.reshape(-1, (rig_highest_dim-5)*rig_numObjs)[0]).repeat(anim_frame_len, 1)
    for entry_index, entry in enumerate(denorm_predict):
        replace_index = 0
        for nan_index, nan in enumerate(rig_nan_tensor[entry_index]):
            if not np.isnan(nan):
                rig_nan_tensor[entry_index][nan_index] = entry[replace_index]
                replace_index += 1
    rig_nan_tensor = rig_nan_tensor.reshape(-1, rig_highest_dim-5)

    # extract values before and after quat
    before_quat_tensor = torch.tensor(rig_nan_tensor[:, :rotMtx_start]).to(device)
    after_quat_tensor = torch.tensor(rig_nan_tensor[:, rotMtx_start+4:]).to(device)
    #after_quat_tensor = torch.tensor(denorm_predict[:, rotMtx_start+6:]).to(device)

    # extract rot matrix and convert to quaternion
    quat_tensor = torch.tensor(rig_nan_tensor[:, rotMtx_start:rotMtx_start+4]).reshape(-1, 4)
    #quat_tensor = torch.tensor(denorm_predict[:, rotMtx_start:rotMtx_start+6])
    rotMtx_tensor = torchUtils.batch_quaternion_to_rotation_matrix(quat_tensor).reshape(-1, 9).to(device)
    #rotMtx_tensor = torchUtils._6d_to_matrix(quat_tensor).reshape(-1, 9).to(device)
    rotMtx_predict = torch.cat((before_quat_tensor, rotMtx_tensor, after_quat_tensor), dim=1)

    predict_rowBegin = train_rig_df.iloc[:rig_numObjs, :3].values
    predict_data = [predict_rowBegin[i%rig_numObjs].tolist() + data.tolist() for i, data in enumerate(rotMtx_predict)]

    print("PROGRESS 90")
    # save anim data for Maya
    predict_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "predict_data/irm_predict_data.csv")
    predict_header = train_rig_df.columns.values # use header of rig train data for predict header

    with open(predict_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(predict_header)
        writer.writerows(predict_data)

    print("PROGRESS 100")