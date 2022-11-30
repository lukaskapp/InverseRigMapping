import subprocess
import sys
import os
import pathlib
import maya.api.OpenMaya as om


def train_model(train_data="cube_data_02.csv"):
    py_path = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
    py_app = pathlib.PurePath(py_path.parent, pathlib.Path("venv/Scripts/python.exe")).as_posix()

    py_cmd = "import sys; sys.path.append('{}'); import gpr; gpr.gpr('{}')".format(py_path, train_data)
    
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