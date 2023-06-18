import maya.cmds as cmds
import pymel.core as pm
import maya.api.OpenMaya as om
import math
import os
import random
import csv
import pathlib
from imp import reload

import utils.maya as mUtils
reload(mUtils)
import utils.data_gen as genUtils
reload(genUtils)
import utils.ui as uiUtils
reload(uiUtils)

def generate_data(rig_input_data, jnt_input_data, rig_out, jnt_out, train_poses):
    # filter selection into joints and controls
    ctrl_list = [ctrl for ctrl in rig_input_data.keys() if mUtils.query_visibility(ctrl)]
    ctrl_list.sort()

    jnt_list = [jnt for jnt in jnt_input_data.keys()]
    jnt_list.sort()

    for ctrl in ctrl_list:
        mUtils.restore_defaults(ctrl)

    ctrl_unique_attrs = list(set([attr for ctrl in ctrl_list for attr in mUtils.filter_attrs_from_dict(rig_input_data[ctrl])]))
    # if one rotation axis is included in "ctrl_unique_attrs" lis, remove all rotation axis (X,Y,Z) and add a single "rotate" value
    # as a hint for later that a rotation matrix should be included in the train data
    if ("rotateX" or "rotateY" or "rotateZ") in ctrl_unique_attrs:
        ctrl_unique_attrs = list(set(ctrl_unique_attrs).difference(["rotateX", "rotateY", "rotateZ"]))
        ctrl_unique_attrs.append("rot_mtx")
    ctrl_unique_attrs.sort() # reoder list to make it independent of control selection

    # list all unique attrs across all joints / controls - that will define the amount of column headers
    jnt_unique_attrs = list(set([jnt_attr[0] for jnt in jnt_list for jnt_attr in jnt_input_data[jnt]]))
    if ("rotateX" or "rotateY" or "rotateZ") in jnt_unique_attrs:
        jnt_unique_attrs = list(set(jnt_unique_attrs).difference(["rotateX", "rotateY", "rotateZ"]))
        jnt_unique_attrs.append("rot_mtx")
    jnt_unique_attrs.sort() # reoder list to make it independent of joint selection

    # build data header based on unique attrs of controls and joints
    rig_header = genUtils.build_header(base_header=["No.", "rigName", "dimension"], attr_list=ctrl_unique_attrs)
    jnt_header = genUtils.build_header(base_header=["No.", "jointName", "dimension"], attr_list=jnt_unique_attrs)

    # create empty csv files, fill them row by row with data
    with open(rig_out, "w") as f:
        writer = csv.writer(f)
        writer.writerow(rig_header)
    
    with open(jnt_out, "w") as f:
        writer = csv.writer(f)
        writer.writerow(jnt_header)

    # Initialize the progress window
    cmds.progressWindow(title='Generating...', progress=0, status='Starting...', isInterruptable=True)

    rig_data = []
    jnt_data = []
    for i in range(train_poses):
        if cmds.progressWindow(query=True, isCancelled=True):
            break
        
        for ctrl_index, ctrl in enumerate(ctrl_list):
            # only get integer and float attributes of selected control
            attr_list = [attr[0] for attr in rig_input_data[ctrl]]
            
            # check if rotation is in attr list
            rotation = genUtils.check_for_rotation(attr_list)

            # set dimension to length of attr_list; if rotate in list, remove rotate and add matrix3 (9 dimension)
            attr_dimension = genUtils.get_attr_dimension(attr_list, rotation)

            input_range_list = {}
            for values in rig_input_data[ctrl]:
                input_range_list[values[0]] = [values[1], values[2]]

            random.seed(i+ctrl_index)
            for attr in attr_list:
                input_rand_min = float(input_range_list[attr][0])
                input_rand_max = float(input_range_list[attr][1])

                if cmds.attributeQuery(attr, node=ctrl, minExists=1) or cmds.attributeQuery(attr, node=ctrl, maxExists=1): # if min or max range exists, use those as new values for rand min/max
                    if cmds.attributeQuery(attr, node=ctrl, minExists=1):
                        rand_min = cmds.attributeQuery(attr, node=ctrl, min=1)[0]
                    else:
                        rand_min = input_rand_min

                    if cmds.attributeQuery(attr, node=ctrl, maxExists=1):
                        rand_max = cmds.attributeQuery(attr, node=ctrl, max=1)[0]
                    else:
                        rand_max = input_rand_max
                else:
                    rand_min, rand_max = mUtils.check_transformLimit(ctrl, axis=attr, default_min=input_rand_min, default_max=input_rand_max)

                if rand_min < input_rand_min:
                    rand_min = input_rand_min

                if rand_max > input_rand_max:
                    rand_max = input_rand_max

                cmds.setAttr("{}.{}".format(ctrl, attr), round(random.uniform(rand_min, rand_max), 5))


            # create list with n/a for every attr in rig_header
            rig_data_row = [ctrl_index, ctrl, attr_dimension]
            rig_data_row.extend(["n/a" for i in range(len(rig_header)-3)])            
            for attr in attr_list:
                if not attr in ["rotateX", "rotateY", "rotateZ"]:
                    # replace only used attr of ctrls in n/a list, rest stays at n/a
                    rig_data_row[rig_header.index(attr)] = cmds.getAttr("{}.{}".format(ctrl, attr))


            if rotation:
                ctrl_mtx = pm.dt.TransformationMatrix(cmds.xform(ctrl, m=1, q=1, os=1))
                ctrl_rot_mtx3 = [x for mtx in ctrl_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]

                start_index = rig_header.index("rotMtx_00") # get index of first rotMtx entry in rig_header and start replacing rotMtx values from there
                for mtx_index, rot_mtx in enumerate(ctrl_rot_mtx3):
                    rig_data_row[start_index + mtx_index] = rot_mtx

            with open(rig_out, "a") as f:
                writer = csv.writer(f)
                writer.writerow(rig_data_row)


        for y, jnt in enumerate(jnt_list):
            attr_list = [attr[0] for attr in jnt_input_data[jnt]]
            rotation = genUtils.check_for_rotation(attr_list)
            jnt_dimension = genUtils.get_attr_dimension(attr_list, rotation)
           
            jnt_data_row = [y, jnt, jnt_dimension]
            jnt_data_row.extend(["n/a" for i in range(len(jnt_header)-3)])   
            for attr in attr_list:
                if not attr in ["rotateX", "rotateY", "rotateZ"]:
                    # replace only used attr of jnts in n/a list, rest stays at n/a
                    jnt_data_row[jnt_header.index(attr)] = cmds.getAttr("{}.{}".format(jnt, attr))

            if rotation:
                jnt_mtx = pm.dt.TransformationMatrix(cmds.xform(jnt, m=1, q=1, os=1))
                jnt_rot_mtx3 = [x for mtx in jnt_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]

                start_index = jnt_header.index("rotMtx_00") # get index of first rotMtx entry in jnt_header and start replacing rotMtx values from there
                for mtx_index, rot_mtx in enumerate(jnt_rot_mtx3):
                    jnt_data_row[start_index + mtx_index] = rot_mtx

            with open(jnt_out, "a") as f:
                writer = csv.writer(f)
                writer.writerow(jnt_data_row)

        #update progress
        cmds.progressWindow(edit=True, progress=((float(i+1))/train_poses)*100,
                            status=f'Generating {i+1}/{train_poses}...')


    # reset transforms to zero
    for ctrl in ctrl_list:
        mUtils.restore_defaults(ctrl)

    cmds.progressWindow(endProgress=True)

