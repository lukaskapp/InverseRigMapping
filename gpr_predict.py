import math
import torch
import gpytorch
import numpy as np
import pandas as pd
import csv
from importlib import reload
import pathlib
import os
import modelList_gpr as gpr
reload(gpr)


# enable GPU/CUDA if available
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

#anim_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "anim_data_01.csv")

def predict_data(anim_path):
    likelihood, model = gpr.gpr()
    likelihood.to(device)
    model.to(device)

    # load anim data
    anim_dataset = pd.read_csv(anim_path, na_values='?', comment='\t', sep=',', skipinitialspace=True, header=[0])

    # get number of objs in first column and mult it with len of data entries
    anim_x_depth =  len(np.unique(anim_dataset.iloc[:, :1]))
    mean_depth = len(anim_dataset.iloc[:, 2:].values[0])
    anim_predict_x = torch.from_numpy(np.array(anim_dataset.iloc[:, 2:]).reshape(anim_x_depth, -1)).float()
    anim_predict_x = anim_predict_x.to(device)

    anim_predict_x_trans = anim_predict_x[:, :3]
    anim_predict_x_rot = anim_predict_x[:, 3:]

    x_trans_means = anim_predict_x_trans.mean(1, keepdim=True)
    x_trans_deviations = anim_predict_x_trans.std(1, keepdim=True)
    #anim_predict_x_trans_norm = (anim_predict_x_trans - x_trans_means) / x_trans_deviations
    #anim_predict_x_trans_norm = torch.nn.functional.normalize(anim_predict_x_trans)
    if (anim_predict_x_trans.max() - anim_predict_x_trans.min()) == 0:
        anim_predict_x_trans_norm = anim_predict_x_trans
    else:
        anim_predict_x_trans_norm = (anim_predict_x_trans - anim_predict_x_trans.min()) / (anim_predict_x_trans.max() - anim_predict_x_trans.min())
    anim_predict_x_rot_norm = (anim_predict_x_rot - anim_predict_x_rot.min()) / (anim_predict_x_rot.max() - anim_predict_x_rot.min())

    anim_predict_x_norm = torch.cat((anim_predict_x_trans_norm, anim_predict_x_rot_norm), 1)

    predict_y = likelihood(*model(*anim_predict_x_norm))
    predict_mean = []
    for predict in predict_y:
        mean = predict.mean.reshape(-1, mean_depth)
        mean.to(device)
        predict_mean.append(mean)

    #predict_mean = torch.stack([predict_y[0].mean.reshape(-1, anim_x_depth), predict_y[1].mean.reshape(-1, anim_x_depth), predict_y[2].mean.reshape(-1, anim_x_depth)])

    predict_dataset = anim_dataset.copy()
    for i, obj in enumerate(predict_mean):
        for row, transforms in enumerate(obj):
            for column, attr in enumerate(transforms):
                rigName = predict_dataset[predict_dataset.columns.values[1]][row+ (len(obj)*i)]
                predict_dataset.at[row+ (len(obj)*i), predict_dataset.columns.values[1]] = rigName.replace("_anim_bind", "_ctrl")

                predict_dataset.at[row+ (len(obj)*i), predict_dataset.columns.values[column+2]] = attr.detach().cpu().numpy()


    # save anim data for Maya
    predict_name = "predict_cube_data.csv"
    predict_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "predict_data", predict_name)

    header = predict_dataset.columns.values
    predict_dataset.to_csv(predict_path, header=header, index=False)


if __name__=="__main__":
    predict_data(anim_path=pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", "anim_data_01.csv"))