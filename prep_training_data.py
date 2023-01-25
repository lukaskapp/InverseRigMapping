import maya.cmds as cmds
import pymel.core as pm
import maya.api.OpenMaya as om
import math
import os
import random
import csv
import pathlib


def prep_data():
    sel = cmds.ls(sl=1)

    if len(sel) == 0:
        om.MGlobal.displayError("Nothing selected! Please select one control and one or more joints!")
        return

    # filter selection into joints and controls
    ctrl_list = [ctrl for ctrl in cmds.ls(sl=1, typ="transform") if "_ctrl" in ctrl and not "_srtBuffer" in ctrl]
    print(ctrl_list)

    jnt_list = [jnt for jnt in cmds.ls(sl=1, typ="joint") if "_bind" in jnt and not "_end_bind" in jnt]
    print(jnt_list)


    rig_data = []
    jnt_data = []
    for i in range(500):
        for x, ctrl in enumerate(ctrl_list):
            random.seed(i+x)
            tx = round(random.uniform(-50, 50), 5)
            ty = round(random.uniform(-50, 50), 5)
            tz = round(random.uniform(-50, 50), 5)
            ctrl_pos = [tx,ty,tz]
            print("CTRL POS: ", ctrl_pos)

            rx = round(random.uniform(-180,180), 5)
            ry = round(random.uniform(-180,180), 5)
            rz = round(random.uniform(-180,180), 5)
            ctrl_rot = [rx, ry, rz]
            
            cmds.xform(ctrl, t=ctrl_pos, os=1)
            cmds.xform(ctrl, ro=ctrl_rot, os=1)


            ctrl_mtx = pm.dt.TransformationMatrix(cmds.xform(ctrl, m=1, q=1, os=1))
            ctrl_rot_mtx3 = [x for mtx in ctrl_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]
            ctrl_trans = ctrl_mtx.getTranslation("object")

            rig_data_add = [x]
            rig_data_add.extend([ctrl, 12, "translateX", ctrl_trans[0], "translateY", ctrl_trans[1], "translateZ", ctrl_trans[2],
                                        "rotate_00", ctrl_rot_mtx3[0], "rotate_01", ctrl_rot_mtx3[1], "rotate_02", ctrl_rot_mtx3[2],
                                        "rotate_10", ctrl_rot_mtx3[3], "rotate_11", ctrl_rot_mtx3[4], "rotate_12", ctrl_rot_mtx3[5],
                                        "rotate_20", ctrl_rot_mtx3[6], "rotate_21", ctrl_rot_mtx3[7], "rotate_22", ctrl_rot_mtx3[8]])

            #rig_data_add.extend([ctrl, 3, "translateX", ctrl_trans[0], "translateY", ctrl_trans[1], "translateZ", ctrl_trans[2]])   

            rig_data.append(rig_data_add)


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
        cmds.xform(ctrl, t=[0,0,0], ro=[0,0,0], s=[1,1,1], os=1)


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