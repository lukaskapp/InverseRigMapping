"""
-----------------------------------------------------------------------------
This file has been developed within the scope of the
Technical Director course at Filmakademie Baden-Wuerttemberg.
http://technicaldirector.de

Written by Lukas Kapp
Copyright (c) 2023 Animationsinstitut of Filmakademie Baden-Wuerttemberg
-----------------------------------------------------------------------------
"""

import maya.cmds as cmds
import pymel.core as pm
import maya.api.OpenMaya as om
import math
import csv
import pathlib
import os
import subprocess

import system.prep_anim_data as prep_anim_data



def get_predict_data(anim_path, model_path, rig_path, py_app):
    py_path = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__)))).parent

    py_cmd = f"import sys; sys.path.append('{py_path}'); import system.gpr_predict as gpr_predict; gpr_predict.predict_data('{anim_path}', '{model_path}', '{rig_path}')"
    
    command = [py_app, "-u", "-c", py_cmd]
    # start subprocess; prevent output window
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)

    # update progress bar

    for line in iter(process.stdout.readline, ""):
        if line.startswith("PROGRESS "):
            progress = float(line.strip().split(" ")[1])
            if cmds.progressWindow(query=True, isCancelled=True):
                process.kill()
                break
            cmds.progressWindow(edit=True, progress=progress, status=(f'Progress: {progress} %'))

    process.wait()

    if process.returncode:
        error_message = process.stderr.read()
        raise ValueError(error_message)
    else:
        om.MGlobal.displayInfo("Data predicted successfully!")



def map_data(anim_input_data, jnt_path, rig_path, model_path, py_app):
    # Initialize the progress window
    cmds.progressWindow(title='Map Prediction', progress=0, status='Getting Animation Data...', isInterruptable=True)

    # get animation data
    anim_path, frames = prep_anim_data.prep_data(anim_input_data, jnt_path)

    # predict data
    cmds.progressWindow(edit=True, progress=0, status=('Initialize Prediction...'))
    get_predict_data(anim_path, model_path, rig_path, py_app)

    cmds.progressWindow(edit=True, progress=0, status=('Applying Prediction...'))
    predict_path = pathlib.PurePath(pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__)))).parent, "predict_data/irm_predict_data.csv")    
    with open(predict_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None) # skip header
        
        predict_data = [row for row in reader if len(row) != 0]
        ctrl_depth = len(list(dict.fromkeys([int(n[0]) for n in predict_data])))

        current_frame = cmds.currentTime(q=1)
        for i, data in enumerate(predict_data):
            ctrl = data[1]
            values = data[3:]
            # predict data is ctrl depth times frames
            # so need to divide i with ctrl depth to get current anim frame
            frame = frames[math.floor(i/ctrl_depth)]
            rotMtx = []
            for value in values:
                if value != "nan":
                    attr_name = header[data.index(value)]
                    if "rotMtx_" in attr_name:
                        rotMtx.append(value)
                        continue
                    cmds.setAttr("{}.{}".format(ctrl, attr_name), float(value))
                    cmds.setKeyframe(ctrl, t=frame, at=attr_name, v=float(value))

            if rotMtx:
                mtx = pm.dt.TransformationMatrix((float(rotMtx[0]), float(rotMtx[1]), float(rotMtx[2]), 0.0,
                                                float(rotMtx[3]), float(rotMtx[4]), float(rotMtx[5]), 0.0,
                                                float(rotMtx[6]), float(rotMtx[7]), float(rotMtx[8]), 0.0,
                                                0.0, 0.0, 0.0, 1.0)).euler
                rot = [math.degrees(mtx[0]), math.degrees(mtx[1]), math.degrees(mtx[2])]
                
                for attr in ["rx", "ry", "rz"]:
                    cmds.xform(ctrl, ro=rot, os=1)
                    value = cmds.getAttr("{}.{}".format(ctrl, attr))
                    cmds.setKeyframe(ctrl, t=frame, at=attr, v=float(value))

                    cmds.xform(ctrl, s=[1.0,1.0,1.0])
                    cmds.setAttr("{}.shearXY".format(ctrl), 0.0)
                    cmds.setAttr("{}.shearXZ".format(ctrl), 0.0)
                    cmds.setAttr("{}.shearYZ".format(ctrl), 0.0)

            cmds.progressWindow(edit=True, progress=(i/len(predict_data))*100, status=('Applying Prediction...'))  
        cmds.currentTime(current_frame)

    cmds.progressWindow(endProgress=True) 
 