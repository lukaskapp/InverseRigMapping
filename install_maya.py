import maya.cmds as cmds
import maya.mel as mel
import os


def create_shelf_button():
    irm_path = os.environ.get('IRM_PATH')
    if irm_path is None:
        print("Environment variable not found.")
    else:
        print(irm_path)

    button_command = f'''
import sys
from imp import reload

path = "{irm_path}

if not path in sys.path:
    sys.path.append(path)

import irm_ui as ui
reload(ui)
    '''

    # Find the currently active shelf
    top_shelf = mel.eval("$tempVar = $gShelfTopLevel")
    active_shelf = cmds.tabLayout(top_shelf, query=True, selectTab=True)

    # Create a new button in the current shelf
    cmds.shelfButton(
        parent=active_shelf,
        command=button_command,
        annotation='IRM Tool',
        image1='commandButton.png',  # Replace with your icon
        width=10,
        height=10,
        label='IRM_button',
        imageOverlayLabel='IRM'
    )

create_shelf_button()
