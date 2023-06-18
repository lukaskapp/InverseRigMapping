import subprocess
import sys
import os
import pathlib
import maya.api.OpenMaya as om


def train_model(py_app, rig_path="irm_rig_data.csv", jnt_path="irm_jnt_data.csv", model_path="trained_model.pt", lr=0.01, epochs=100, force_cpu=False):
    py_path = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
    py_cmd = f"import sys; sys.path.append('{py_path}'); import gpr_model as gpr; gpr.train_model('{rig_path}', '{jnt_path}', '{model_path}', {lr}, {epochs}, {force_cpu})"
    
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


['R:/diploma_inverse_rig_mapping/venv/Scripts/python.exe', '-c', "import sys; sys.path.append('R:\\diploma_inverse_rig_mapping\\code_inverseRigMapping'); import gpr_model as gpr; gpr.train_model('R:\\diploma_inverse_rig_mapping\\code_inverseRigMapping\\training_data\\rig\\irm_rig_data.csv', 'R:/diploma_inverse_rig_mapping/code_inverseRigMapping/training_data/jnt/irm_jnt_data.csv', 'R:/diploma_inverse_rig_mapping/code_inverseRigMapping/trained_model/trained_model.pt', 0.01, 100, False)"]
