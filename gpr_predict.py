import math
import torch
import gpytorch
import numpy as np
import pandas as pd
import csv
from importlib import reload
import pathlib
import os
#import modelList_gpr as gpr
import multiTask_gpr as gpr
reload(gpr)


# enable GPU/CUDA if available
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

#anim_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "anim_data_01.csv")

def predict_data(anim_path):
    likelihood, model, x_trans_min, x_trans_max = gpr.gpr()
    likelihood.to(device)
    model.to(device)

    # load anim data
    anim_dataset = pd.read_csv(anim_path, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    # get number of objs in first column and mult it with len of data entries
    anim_x_depth =  len(np.unique(anim_dataset.iloc[:, :1]))
    mean_depth = len(anim_dataset.iloc[:, 2:].values[0])
    anim_x = torch.from_numpy(np.array(anim_dataset.iloc[:, 2:]).reshape(-1, mean_depth)).float()
    anim_x = anim_x.to(device)

    anim_x_trans = anim_x[:, :3]
    anim_x_rot = anim_x[:, 3:]

    # normalize data to range -1.0 1.0
    new_min, new_max = -1.0, 1.0

    #x_trans_min, x_trans_max = -40.0, 40.0
    anim_x_trans_norm = (anim_x_trans - x_trans_min) / (x_trans_max - x_trans_min) * (new_max - new_min) + new_min

    #x_rot_min, x_rot_max = anim_x_rot.min(), anim_x_rot.max()
    #anim_x_rot_norm = (anim_x_rot - x_rot_min) / (x_rot_max - x_rot_min) * (new_max - new_min) + new_min


    anim_x_norm = torch.cat((anim_x_trans_norm, anim_x_rot), -1).reshape(-1, mean_depth)
    #anim_x_norm = anim_x



    predict_y = likelihood(model(anim_x_norm))
    predict_mean = predict_y.mean
    #predict_mean = []
    #for predict in predict_y:
    #    mean = predict.mean.reshape(-1, mean_depth)
    #    mean.to(device)
    #    predict_mean.append(mean)

    """
    predict_dataset = anim_dataset.copy()
    #for i, obj in enumerate(predict_mean):
    i = 0
    for row, transforms in enumerate(predict_mean):
        for column, attr in enumerate(transforms):
            rigName = predict_dataset[predict_dataset.columns.values[1]][row+ (len(predict_mean)*i)]
            predict_dataset.at[row+ (len(predict_mean)*i), predict_dataset.columns.values[1]] = rigName.replace("_anim_bind", "_ctrl")
            predict_dataset.at[row+ (len(predict_mean)*i), predict_dataset.columns.values[column+2]] = attr.detach().cpu().numpy()
    """





    # save anim data for Maya
    predict_name = "predict_cube_data.csv"
    predict_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "predict_data", predict_name)

    #header = predict_dataset.columns.values
    #predict_dataset.to_csv(predict_path, header=header, index=False)


    predict_data = []
    for translate in predict_mean:
        predict_data.append([0, "irm_C_ik_ctrl", translate[0].cpu().detach().numpy(), translate[1].cpu().detach().numpy(), translate[2].cpu().detach().numpy()])


    header = ["No.", "jointName", "translateX", "translateY", "translateZ"]

    with open(predict_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(predict_data)

#if __name__=="__main__":
#    predict_data(anim_path=pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "anim_data_01.csv"))