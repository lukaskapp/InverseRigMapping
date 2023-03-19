import maya.cmds as cmds
import pymel.core as pm
import maya.api.OpenMaya as om
import math
import csv
import prep_anim_data
from importlib import reload
import pathlib
import os
import subprocess


reload(prep_anim_data)


def chunks(lst, n):
    """split list into n-sized chunks"""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def get_predict_data(anim_path):
    py_path = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
    py_app = pathlib.PurePath(py_path.parent, pathlib.Path("venv/Scripts/python.exe")).as_posix()

    py_cmd = "import sys; sys.path.append('{}'); import gpr_predict; gpr_predict.predict_data('{}')".format(py_path, anim_path)
    
    command = [py_app, "-c", py_cmd]
    process = subprocess.run(command, capture_output=True, text=True)

    if process.returncode:
        raise ValueError(process.stderr)
    else:
        print(process.stdout)
        om.MGlobal.displayInfo("Data predicted successfully!")

        return process.stdout.rpartition("PREDICT DATA")[2]


def map_data():
    anim_path, frames = prep_anim_data.prep_data()
    get_predict_data(anim_path)

    predict_name = "irm_predict_data.csv"
    predict_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "predict_data", predict_name)

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
                
        cmds.currentTime(current_frame)
            
    
#if __name__ == "__main__":
#    map_data()