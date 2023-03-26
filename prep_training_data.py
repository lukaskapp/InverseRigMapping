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

def prep_data(rig_input_data, jnt_input_data):
    if len(rig_input_data) == 0 or len(jnt_input_data) == 0:
        om.MGlobal.displayError("Nothing selected! Please select one control and one or more joints!")
        return

    # filter selection into joints and controls
    ctrl_list = [ctrl for ctrl in rig_input_data if mUtils.query_visibility(ctrl)]
    ctrl_list.sort()

    jnt_list = [jnt for jnt in jnt_input_data]
    jnt_list.sort()


    for ctrl in ctrl_list:
        mUtils.restore_defaults(ctrl)


    ctrl_unique_attrs = list(set([unique_attr for ctrl in ctrl_list for unique_attr in mUtils.get_all_attributes(ctrl)]))
    # if one rotation axis is included in "ctrl_unique_attrs" lis, remove all rotation axis (X,Y,Z) and add a single "rotate" value
    # as a hint for later that a rotation matrix should be included in the train data
    if ("rotateX" or "rotateY" or "rotateZ") in ctrl_unique_attrs:
        ctrl_unique_attrs = list(set(ctrl_unique_attrs).difference(["rotateX", "rotateY", "rotateZ"]))
        ctrl_unique_attrs.append("rot_mtx")
    ctrl_unique_attrs.sort() # reoder list to make it independent of control selection
    #print("CTRL UNIQUE", ctrl_unique_attrs)


    # filter attributes that have incoming connections
    # and store the default values for later use
    jnt_defaults = {}
    for jnt in jnt_list:
        attr_defaults = {}
        for attr in mUtils.get_all_attributes(jnt, unlocked=False):
            if mUtils.check_source_connection(jnt, attr):
                attr_defaults[attr] = cmds.getAttr("{}.{}".format(jnt, attr))
        jnt_defaults[jnt] = attr_defaults
    #print("JNTDEFAULTS", jnt_defaults)

    # list all unique attrs across all joints / controls - that will define the amount of column headers
    jnt_unique_attrs = list(set([jnt_attr for jnt_default in jnt_defaults.values() for jnt_attr in jnt_default.keys()]))
    if ("rotateX" or "rotateY" or "rotateZ") in jnt_unique_attrs:
        jnt_unique_attrs = list(set(jnt_unique_attrs).difference(["rotateX", "rotateY", "rotateZ"]))
        jnt_unique_attrs.append("rot_mtx")
    jnt_unique_attrs.sort() # reoder list to make it independent of joint selection
    #print("JNT UNIQUE", jnt_unique_attrs)

    
    # build data header based on unique attrs of controls and joints
    rig_header = genUtils.build_header(base_header=["No.", "rigName", "dimension"], attr_list=ctrl_unique_attrs)
    jnt_header = genUtils.build_header(base_header=["No.", "jointName", "dimension"], attr_list=jnt_unique_attrs)


    #print("")
    #print("-----------------------------------------------------------")
    #print("")

    rig_data = []
    jnt_data = []

    iterations = 2000
    for i in range(iterations):
        #print("Progress: {}/{}".format(str(i+1).zfill(4), iterations))
        print("Progress: {}%".format(round(((float(i+1))/iterations)*100, 2)))
        for ctrl_index, ctrl in enumerate(ctrl_list):
            # only get integer and float attributes of selected control
            attr_list = mUtils.get_all_attributes(ctrl)
            #print("ATTR LIST: ", attr_list)
            
            
            # check if rotation is in attr list
            rotation = genUtils.check_for_rotation(attr_list)
            #print("ROT: ", rotation)

            # set dimension to length of attr_list; if rotate in list, remove rotate and add matrix3 (9 dimension)
            attr_dimension = genUtils.get_attr_dimension(attr_list, rotation)

            
            #print("DIMENSION: ", attr_dimension)
            #print("-----------------------------------------------------------")


            #default_rand_min, default_rand_max = -170.0, 35.0
            #default_rand_min, default_rand_max = -50.0, 220.0
            default_rand_min, default_rand_max = -50.0, 50.0
            random.seed(i+ctrl_index)
            for attr in attr_list:
                if cmds.attributeQuery(attr, node=ctrl, minExists=1) or cmds.attributeQuery(attr, node=ctrl, maxExists=1): # if min or max range exists, use those as new values for rand min/max
                    if cmds.attributeQuery(attr, node=ctrl, minExists=1):
                        rand_min = cmds.attributeQuery(attr, node=ctrl, min=1)[0]
                    else:
                        rand_min = default_rand_min

                    if cmds.attributeQuery(attr, node=ctrl, maxExists=1):
                        rand_max = cmds.attributeQuery(attr, node=ctrl, max=1)[0]
                    else:
                        rand_max = default_rand_max

                elif attr in ["rotateX", "rotateY", "rotateZ"]: # if rotation, translate or scale is used, check limits and use those for min/max range
                    rand_min, rand_max = mUtils.check_transformLimit(ctrl, axis=attr, default_min=-180, default_max=180)
                elif attr in ["translateX", "translateY", "translateZ"]:
                    rand_min, rand_max = mUtils.check_transformLimit(ctrl, axis=attr, default_min=default_rand_min, default_max=default_rand_max)
                elif attr in ["scaleX", "scaleY", "scaleZ"]:
                    rand_min, rand_max = mUtils.check_transformLimit(ctrl, axis=attr, default_min=0.01, default_max=10)

                #print("ATTR: ", attr)
                #print("RAND MIN: ", rand_min)
                #print("RAND MAX: ", rand_max)
                #print("------------------------")
                cmds.setAttr("{}.{}".format(ctrl, attr), round(random.uniform(rand_min, rand_max), 5))



            # create list with n/a for every attr in rig_header
            rig_data_add = [ctrl_index, ctrl, attr_dimension]
            rig_data_add.extend(["n/a" for i in range(len(rig_header)-3)])            
            for attr in attr_list:
                if not attr in ["rotateX", "rotateY", "rotateZ"]:
                    # replace only used attr of ctrls in n/a list, rest stays at n/a
                    rig_data_add[rig_header.index(attr)] = cmds.getAttr("{}.{}".format(ctrl, attr))


            if rotation:
                ctrl_mtx = pm.dt.TransformationMatrix(cmds.xform(ctrl, m=1, q=1, os=1))
                ctrl_rot_mtx3 = [x for mtx in ctrl_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]

                start_index = rig_header.index("rotMtx_00") # get index of first rotMtx entry in rig_header and start replacing rotMtx values from there
                for mtx_index, rot_mtx in enumerate(ctrl_rot_mtx3):
                    rig_data_add[start_index + mtx_index] = rot_mtx


            rig_data.append(rig_data_add)
            #print("RIG DATA ADD: ", rig_data_add)

            #print("")
            #print("-----------------------------------------------------------")
            #print("")

        for y, jnt in enumerate(jnt_list):
            attr_list = [attr for attr in mUtils.get_all_attributes(jnt, unlocked=False) if mUtils.check_source_connection(jnt, attr)]
            
            rotation = genUtils.check_for_rotation(attr_list)
            jnt_dimension = genUtils.get_attr_dimension(attr_list, rotation)
            
            jnt_data_add = [y, jnt, jnt_dimension]
            jnt_data_add.extend(["n/a" for i in range(len(jnt_header)-3)])   
            for attr in attr_list:
                if not attr in ["rotateX", "rotateY", "rotateZ"]:
                    # replace only used attr of jnts in n/a list, rest stays at n/a
                    jnt_data_add[jnt_header.index(attr)] = cmds.getAttr("{}.{}".format(jnt, attr))



            if rotation:
                jnt_mtx = pm.dt.TransformationMatrix(cmds.xform(jnt, m=1, q=1, os=1))
                jnt_rot_mtx3 = [x for mtx in jnt_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]

                start_index = jnt_header.index("rotMtx_00") # get index of first rotMtx entry in jnt_header and start replacing rotMtx values from there
                for mtx_index, rot_mtx in enumerate(jnt_rot_mtx3):
                    jnt_data_add[start_index + mtx_index] = rot_mtx

            jnt_data.append(jnt_data_add)



    # reset transforms to zero
    for ctrl in ctrl_list:
        mUtils.restore_defaults(ctrl)


    print("RIG DATA: ", rig_data)
    print("JNT DATA: ", jnt_data)

    rig_fileName = "irm_rig_data.csv"
    jnt_fileName = "irm_jnt_data.csv"
    rig_fullpath = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig", rig_fileName)
    jnt_fullpath = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt", jnt_fileName)

    with open(rig_fullpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(rig_header)
        writer.writerows(rig_data)
    
    with open(jnt_fullpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(jnt_header)
        writer.writerows(jnt_data)




if __name__ == "__main__":
    prep_data()