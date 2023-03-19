import math
import torch
import gpytorch
import numpy as np
import pandas as pd
import csv
from importlib import reload
import pathlib
import os
import multiTask_gpr as gpr
reload(gpr)


# enable GPU/CUDA if available
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu"
dev = "cpu" 
device = torch.device(dev)


#anim_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "irm_anim_data.csv")


def predict_data(anim_path):
    # load trained model
    rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", "irm_rig_data.csv")
    jnt_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", "irm_jnt_data.csv")
    train_x = gpr.build_train_x_tensor(jnt_file)
    train_y, train_y_dimension = gpr.build_train_y_tensor(rig_file)

    #likelihood, model, x_trans_min, x_trans_max = gpr.gpr()
    model_file = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "trained_model.pt")
    state_dict = torch.load(model_file)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y_dimension)
    likelihood.to(device)

    model = gpr.MultitaskGPModel(train_x, train_y, likelihood, train_y_dimension)
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    likelihood.eval()



    # load anim data
    anim_dataset = pd.read_csv(anim_path, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    # get number of objs in first column and mult it with len of data entries
    anim_x_depth =  len(np.unique(anim_dataset.iloc[:, :1]))
    anim_data_frames = int(len(anim_dataset)/anim_x_depth)

    mean_depth = len(anim_dataset.iloc[:, 2:].values[0])


    anim_x_raw = np.array(anim_dataset.iloc[:, 2:]).reshape(anim_x_depth, -1, mean_depth)
    anim_x1 = torch.from_numpy(anim_x_raw[0]).float()
    anim_x2 = torch.from_numpy(anim_x_raw[1]).float()
    anim_x3 = torch.from_numpy(anim_x_raw[2]).float()

    anim_x = torch.cat((anim_x1, anim_x2, anim_x3), -1).float()
    #anim_x = torch.from_numpy(np.array(anim_dataset.iloc[:, 2:]).reshape(anim_x_depth, -1)).float()
    anim_x = anim_x.to(device)

    anim_x_trans = anim_x[:, :3]
    anim_x_rot = anim_x[:, 3:]

    # normalize data to range -1.0 1.0
    new_min, new_max = -1.0, 1.0

    #x_trans_min, x_trans_max = -40.0, 40.0
    #anim_x_trans_norm = (anim_x_trans - x_trans_min) / (x_trans_max - x_trans_min) * (new_max - new_min) + new_min

    #x_rot_min, x_rot_max = anim_x_rot.min(), anim_x_rot.max()
    #anim_x_rot_norm = (anim_x_rot - x_rot_min) / (x_rot_max - x_rot_min) * (new_max - new_min) + new_min


    #anim_x_norm = torch.cat((anim_x_trans_norm, anim_x_rot), -1).reshape(-1, mean_depth)
    anim_x_norm = anim_x


    # get predict values from trained model
    predict_y = likelihood(model(anim_x_norm))
    predict_mean = predict_y.mean

    # get rig dataset used in training for building predict data
    rig_file = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", "irm_rig_data.csv")
    train_rig_df = pd.read_csv(rig_file, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    rig_obj_names = np.unique(train_rig_df.iloc[:, 1]) # get names of rig objects for data mapping later
    rig_dimension_list = train_rig_df.filter(items=["dimension"]).values[:len(rig_obj_names)] # get dimension per rig object for data mapping

    predict_data = []
    for data in predict_mean:
        start_range = 0
        # reshape data per frame into dimensions of each rig object as in train rig data
        for i, dimension in enumerate(rig_dimension_list):
            # isolate predict values of single obj
            dimension = int(dimension)
            predict_values = data[start_range:(start_range + dimension)].tolist()
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



#if __name__=="__main__":
#    predict_data(anim_path=pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "anim_data_01.csv"))