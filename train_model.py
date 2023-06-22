import subprocess
import sys
import os
import pathlib
import maya.api.OpenMaya as om
import maya.cmds as cmds


def train_model(py_app, rig_path, jnt_path, model_path, lr=0.01, epochs=100, force_cpu=False):
    py_path = pathlib.Path(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
    py_cmd = f"import sys; sys.path.append('{py_path}'); import gpr_model as gpr; gpr.train_model('{rig_path}', '{jnt_path}', '{model_path}', {lr}, {epochs}, {force_cpu})"
    command = [py_app, "-u", "-c", py_cmd]

    # Initialize the progress window
    cmds.progressWindow(title='Training Model', progress=0, status='Initialize Training...', isInterruptable=True)

    # start subprocess; prevent output window
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)

    # update progress bar
    for line in iter(process.stdout.readline, ""):
        if line.startswith("Iter "):
            current_epoch = str(line.split("/")[0].partition("Iter ")[2])
            current_loss = str(line.split("Loss: ")[1])

        if line.startswith("PROGRESS "):
            progress = float(line.strip().split(" ")[1])
            if cmds.progressWindow(query=True, isCancelled=True):
                process.kill()
                break
            cmds.progressWindow(edit=True, progress=progress, status=(f'Epochs: {current_epoch}/{epochs}   -   Loss: {current_loss}'))

    process.wait()

    cmds.progressWindow(endProgress=True)

    if process.returncode:
        error_message = process.stderr.read()
        raise ValueError(error_message)
    else:
        om.MGlobal.displayInfo(f"Model trained successfully in {epochs} epochs and end loss of {current_loss}")
