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

    predict_name = "predict_cube_data.csv"
    predict_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "predict_data", predict_name)

    with open(predict_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None) # skip header
        
        predict_data = [row for row in reader if len(row) != 0]
        print(predict_data)
        for x in predict_data:
            print(x)

        ctrl_depth = list(dict.fromkeys([int(n[0]) for n in predict_data]))

        values_list = []
        for n in ctrl_depth:
            values_list.append([values for values in predict_data if int(values[0]) == n])
            
        current_frame = cmds.currentTime(q=1)
        
        for i, frame in enumerate(frames):
            for n in ctrl_depth:                        
                print(n)
                ctrl = values_list[n][i][1]
                cmds.currentTime(frame)
                for x, attr in enumerate(header[2:5]):
                    print("SET:  ", "{}.{}".format(ctrl, attr), float(values_list[n][i][2+x]))
                    cmds.setAttr("{}.{}".format(ctrl, attr), float(values_list[n][i][2+x]))
                    cmds.setKeyframe(ctrl, t=frame, at=attr, v=float(values_list[n][i][2+x]))

                    
                """
                offset = 5
                mtx = pm.dt.TransformationMatrix((float(values_list[n][i][offset+0]), float(values_list[n][i][offset+1]), float(values_list[n][i][offset+2]), 0.0,
                                                float(values_list[n][i][offset+3]), float(values_list[n][i][offset+4]), float(values_list[n][i][offset+5]), 0.0,
                                                float(values_list[n][i][offset+6]), float(values_list[n][i][offset+7]), float(values_list[n][i][offset+8]), 0.0,
                                                0.0, 0.0, 0.0, 1.0)).euler
                rot = [math.degrees(mtx[0]), math.degrees(mtx[1]), math.degrees(mtx[2])]
                
                for attr in ["rx", "ry", "rz"]:
                    cmds.xform(ctrl, ro=rot, os=1)
                    #cmds.xform(ctrl, m=mtx.__melobject__(), os=1) # will override translation values due to 0 values in mtx
                    value = cmds.getAttr("{}.{}".format(ctrl, attr))
                    cmds.setKeyframe(ctrl, t=frame, at=attr, v=float(value))

                    cmds.xform(ctrl, s=[1.0,1.0,1.0])
                    cmds.setAttr("{}.shearXY".format(ctrl), 0.0)
                    cmds.setAttr("{}.shearXZ".format(ctrl), 0.0)
                    cmds.setAttr("{}.shearYZ".format(ctrl), 0.0)
                """

        cmds.currentTime(current_frame)
        

    
if __name__ == "__main__":
    map_data()