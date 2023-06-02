import subprocess
import sys
import os
import pathlib
import maya.api.OpenMaya as om


def train_model(rig_fileName="irm_rig_data.csv", jnt_fileName="irm_jnt_data.csv",  model_file="trained_model.pt"):
    py_path = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
    py_app = pathlib.PurePath(py_path.parent, pathlib.Path("venv/Scripts/python.exe")).as_posix()

    py_cmd = "import sys; sys.path.append('{}'); import gpr_model as gpr; gpr.train_model()".format(py_path)
    
    command = [py_app, "-c", py_cmd]
    print(command)

    process = subprocess.run(command, capture_output=True, text=True)

    if process.returncode:
        raise ValueError(process.stderr)
    else:
        print(process.stdout)
        om.MGlobal.displayInfo("Model trained successfully!")

if __name__ == "__main__":
    train_model()