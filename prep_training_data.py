import maya.cmds as cmds
import pymel.core as pm
import maya.api.OpenMaya as om
import math
import os
import random
import csv
import pathlib


def get_all_attributes(obj):
    '''
    return all attributes of obj if attribute is unlocked, keyable, scalar and not of type enum or bool
    '''
    return [attr for attr in cmds.listAttr(obj, unlocked=True, keyable=True, scalar=True) if not cmds.attributeQuery(attr, node=obj, at=1) in ["enum", "bool"]]

def restore_defaults(ctrl):
    '''
    set all attributes on obj to default values
    '''
    for attr in get_all_attributes(ctrl):
        cmds.setAttr("{}.{}".format(ctrl, attr), cmds.attributeQuery(attr, node=ctrl, ld=1)[0])


def check_source_connection(obj, attr):
    connection = cmds.listConnections("{}.{}".format(obj, attr), s=1, d=0, p=1)
    
    # check if attr is part of a compound one and check the compound one as well for any connections
    if not connection:
        attr_parent = cmds.attributeQuery(attr, node=obj, lp=1)[0]
        if attr_parent:
            connection = cmds.listConnections("{}.{}".format(obj, attr_parent), s=1, d=0, p=1)
        
    return connection


def prep_data():
    if len(cmds.ls(sl=1)) == 0:
        om.MGlobal.displayError("Nothing selected! Please select one control and one or more joints!")
        return

    # filter selection into joints and controls
    ctrl_list = [ctrl for ctrl in cmds.ls(sl=1, typ="transform") if "_ctrl" in ctrl and not "_srtBuffer" in ctrl]
    print("CTRL LIST: ", ctrl_list)

    jnt_list = [jnt for jnt in cmds.ls(sl=1, typ="joint") if "_bind" in jnt and not "_end_bind" in jnt]
    print("JNT LIST: ", jnt_list)


    rig_data = []
    jnt_data = []

    for ctrl in ctrl_list:
        restore_defaults(ctrl)

    # filter  attributes that have incoming connections
    # and store the default values for later use
    jnt_defaults = {}
    for jnt in jnt_list:
        attr_defaults = {}
        for attr in get_all_attributes(jnt):
            if check_source_connection(jnt, attr):
                attr_defaults[attr] = cmds.getAttr("{}.{}".format(jnt, attr))
        jnt_defaults[jnt] = attr_defaults
    print(jnt_defaults)


    for i in range(500):
        for x, ctrl in enumerate(ctrl_list):
            # only get integer and float attributes
            attr_list = get_all_attributes(ctrl)
            
            # check if rotation is in attr list
            rotation = [rot for rot in attr_list if "rotate" in attr]
            rotation = True

            # set dimension to length of attr_list; if rotate in list, remove rotate and add matrix3 (9 dimension)
            if rotation:
                attr_dimension = len([attr for attr in attr_list if not "rotate" in attr]) + 9
            else:
                attr_dimension = len(attr_list)


            rand_min = -50
            rand_max = 50

            random.seed(i+x)
            for attr in attr_list:
                if "translate" in attr:
                    rand_min = -50
                elif "rotate" in attr:
                    rand_min = -180
                cmds.setAttr("{}.{}".format(ctrl, attr), round(random.uniform(rand_min, -rand_min), 5))


            #if "translate" in attr_list:
            #    ctrl_trans = ctrl_mtx.getTranslation("object")

            rig_data_add = [x, ctrl, attr_dimension]
            for attr in attr_list:
                if "rotate" in attr:
                    continue
                value = cmds.getAttr("{}.{}".format(ctrl, attr))
                rig_data_add.extend([attr, value])

            if rotation:
                ctrl_mtx = pm.dt.TransformationMatrix(cmds.xform(ctrl, m=1, q=1, os=1))
                ctrl_rot_mtx3 = [x for mtx in ctrl_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]

                rig_data_add.extend(["rotate_00", ctrl_rot_mtx3[0], "rotate_01", ctrl_rot_mtx3[1], "rotate_02", ctrl_rot_mtx3[2],
                                    "rotate_10", ctrl_rot_mtx3[3], "rotate_11", ctrl_rot_mtx3[4], "rotate_12", ctrl_rot_mtx3[5],
                                    "rotate_20", ctrl_rot_mtx3[6], "rotate_21", ctrl_rot_mtx3[7], "rotate_22", ctrl_rot_mtx3[8]])

            rig_data.append(rig_data_add)
            print(rig_data_add)


        for y, jnt in enumerate(jnt_list):
            jnt_rot = [round(rot, 3) for rot in cmds.xform(jnt, q=1, ro=1, os=1)]

            jnt_mtx = pm.dt.TransformationMatrix(cmds.xform(jnt, m=1, q=1, os=1))
            jnt_rot_mtx3 = [x for mtx in jnt_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]
            jnt_trans = jnt_mtx.getTranslation("object")

            print("JNT POS: ", jnt_trans)
            print("JNT ROT: ", jnt_rot)

            
            jnt_data_add = [y]
            #jnt_data_add.extend([jnt, jnt_trans[0], jnt_trans[1], jnt_trans[2],
            #                                                    jnt_rot_mtx3[0], jnt_rot_mtx3[1], jnt_rot_mtx3[2],
            #                                                    jnt_rot_mtx3[3], jnt_rot_mtx3[4], jnt_rot_mtx3[5],
            #                                                    jnt_rot_mtx3[6], jnt_rot_mtx3[7], jnt_rot_mtx3[8]])

            jnt_data_add.extend([jnt, 
                                                                jnt_rot_mtx3[0], jnt_rot_mtx3[1], jnt_rot_mtx3[2],
                                                                jnt_rot_mtx3[3], jnt_rot_mtx3[4], jnt_rot_mtx3[5],
                                                                jnt_rot_mtx3[6], jnt_rot_mtx3[7], jnt_rot_mtx3[8]])

            jnt_data.append(jnt_data_add)



    # reset transforms to zero
    for ctrl in ctrl_list:
        restore_defaults(ctrl)

    rig_header = ["No.", "rigName", "dimension", "translateX", "translateX_value", "translateY", "translateY_value", "translateZ", "translateZ_value",
                                        "rotate_00", "rotate_00_value", "rotate_01", "rotate_01_value", "rotate_02", "rotate_02_value",
                                        "rotate_10", "rotate_10_value", "rotate_11", "rotate_11_value", "rotate_12", "rotate_12_value",
                                        "rotate_20", "rotate_20_value", "rotate_21", "rotate_21_value", "rotate_22", "rotate_22_value"]

    #rig_header = ["No.", "rigName", "dimension", "translateX", "translateX_value", "translateY", "translateY_value", "translateZ", "translateZ_value"]


    #jnt_header = ["No.", "jointName", "translateX", "translateY", "translateZ",
    #                                "rotate_00", "rotate_01", "rotate_02",
    #                                "rotate_10", "rotate_11", "rotate_12",
    #                                "rotate_20", "rotate_21", "rotate_22"]

    jnt_header = ["No.", "jointName",
                                    "rotate_00", "rotate_01", "rotate_02",
                                    "rotate_10", "rotate_11", "rotate_12",
                                    "rotate_20", "rotate_21", "rotate_22"]

    print("RIG DATA: ", rig_data)
    print("JNT DATA: ", jnt_data)

    rig_fileName = "cube_rig_data_05.csv"
    jnt_fileName = "cube_jnt_data_05.csv"
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