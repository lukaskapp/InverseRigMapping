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

    sel_ctrl = sel[0]
    sel_jnt = sel[1]

    if cmds.listRelatives(sel_ctrl, ad=1, typ="transform"):
        ctrl_list = [obj for obj in cmds.listRelatives(sel_ctrl, ad=1, typ="transform") if "_ctrl" in obj and not "_srtBuffer" in obj]
        ctrl_list.append(sel_ctrl)
        ctrl_list.reverse()
        print(ctrl_list)
    else:
        ctrl_list = [sel_ctrl]

    if cmds.listRelatives(sel_ctrl, ad=1, typ="joint"):
        jnt_list = [obj for obj in cmds.listRelatives(sel_jnt, ad=1, typ="joint") if "_bind" in obj and not "_end_bind" in obj]
        jnt_list.append(sel_jnt)
        jnt_list.reverse()
        print(jnt_list)
    else:
        jnt_list = [sel_jnt]

    rig_data = []
    jnt_data = []
    for i in range(100):
        for x, ctrl in enumerate(ctrl_list):
            random.seed(i+x)
            tx = round(random.uniform(-250, 250), 2)
            ty = round(random.uniform(-250, 250), 2)
            tz = round(random.uniform(-250, 250), 2)
            ctrl_pos = [tx,ty,tz]

            rx = round(random.uniform(-180,180), 2)
            ry = round(random.uniform(-180,180), 2)
            rz = round(random.uniform(-180,180), 2)
            #ctrl_rot = om.MEulerRotation(math.radians(rx) ,math.radians(ry), math.radians(rz), om.MEulerRotation.kXYZ)
            ctrl_rot = [rx, ry, rz]
            


            #sx = round(random.uniform(-45, 45), 4)
            #sy = round(random.uniform(-45, 45), 4)
            #sz = round(random.uniform(-45, 45), 4)
            #ctrl_scale = [sx,sy,sz]

            #cmds.xform(ctrl, t=ctrl_pos, ro=ctrl_rot, s=ctrl_scale, ws=1)
            cmds.xform(ctrl,  t=ctrl_pos, ro=ctrl_rot, os=1)


            ctrl_mtx = pm.dt.TransformationMatrix(cmds.xform(ctrl, m=1, q=1, os=1)).asRotateMatrix()[:-1]
            ctrl_mtx3 = [x for mtx in ctrl_mtx for x in mtx[:-1]]

            jnt_pos = [round(pos, 3) for pos in cmds.xform(ctrl.replace("_ctrl", "_bind"), q=1, t=1, os=1)]

            #jnt_rot = [round(rot, 5) for jnt in jnt_list for rot in om.MEulerRotation(math.radians(cmds.xform(jnt, q=1, ro=1, ws=1)[0]) ,math.radians(cmds.xform(jnt, q=1, ro=1, ws=1)[1]), math.radians(cmds.xform(jnt, q=1, ro=1, ws=1)[2]), om.MEulerRotation.kXYZ).asQuaternion()]
            jnt_rot = [round(rot, 3) for rot in cmds.xform(ctrl.replace("_ctrl", "_bind"), q=1, ro=1, os=1)]

            jnt_mtx = pm.dt.TransformationMatrix(cmds.xform(ctrl.replace("_ctrl", "_bind"), m=1, q=1, os=1)).asRotateMatrix()[:-1]
            jnt_mtx3 = [x for mtx in jnt_mtx for x in mtx[:-1]]
            #jnt_scale = [scale for jnt in jnt_list for scale in cmds.xform(jnt, q=1, s=1, r=1)]

            cmds.xform(ctrl, t=[0,0,0], ro=[0,0,0], s=[1,1,1], os=1)

            print("JNT POS: ", jnt_pos)
            print("JNT ROT: ", jnt_rot)
            #print("JNT SCALE: ", jnt_scale)
            #ctrl_rot = ctrl_rot.asQuaternion()

            rig_data_add = [x]
            #rig_data_add.extend([sel[0], 9, "translateX", ctrl_pos[0], "translateY", ctrl_pos[1], "translateZ", ctrl_pos[2], "rotateX", ctrl_rot[0], "rotateY", ctrl_rot[1], "rotateZ", ctrl_rot[2], "scaleX", ctrl_scale[0], "scaleY", ctrl_scale[1], "scaleZ", ctrl_scale[2]])
            #rig_data_add.extend([ctrl, 6, "translateX", ctrl_pos[0], "translateY", ctrl_pos[1], "translateZ", ctrl_pos[2], "rotateX", ctrl_rot[0], "rotateY", ctrl_rot[1], "rotateZ", ctrl_rot[2]])
            #rig_data_add.extend([ctrl, 3, "rotateX", ctrl_rot[0], "rotateY", ctrl_rot[1], "rotateZ", ctrl_rot[2]])
            rig_data_add.extend([ctrl, 12, "translateX", ctrl_pos[0], "translateY", ctrl_pos[1], "translateZ", ctrl_pos[2],
                                        "rotate_00", ctrl_mtx3[0], "rotate_01", ctrl_mtx3[1], "rotate_02", ctrl_mtx3[2],
                                        "rotate_10", ctrl_mtx3[3], "rotate_11", ctrl_mtx3[4], "rotate_12", ctrl_mtx3[5],
                                        "rotate_20", ctrl_mtx3[6], "rotate_21", ctrl_mtx3[7], "rotate_22", ctrl_mtx3[8]])
            rig_data.append(rig_data_add)
            
            jnt_data_add = [x]
            #jnt_data_add.extend([jnt_list[x], jnt_pos[0], jnt_pos[1], jnt_pos[2], jnt_rot[0], jnt_rot[1], jnt_rot[2], jnt_scale[0], jnt_scale[1], jnt_scale[2]])
            #jnt_data_add.extend([ctrl.replace("_ctrl", "_bind"), jnt_pos[0], jnt_pos[1], jnt_pos[2], jnt_rot[0], jnt_rot[1], jnt_rot[2]])
            #jnt_data_add.extend([ctrl.replace("_ctrl", "_bind"), jnt_rot[0], jnt_rot[1], jnt_rot[2]])
            jnt_data_add.extend([ctrl.replace("_ctrl", "_bind"), jnt_pos[0], jnt_pos[1], jnt_pos[2],
                                                                jnt_mtx3[0], jnt_mtx3[1], jnt_mtx3[2],
                                                                jnt_mtx3[3], jnt_mtx3[4], jnt_mtx3[5],
                                                                jnt_mtx3[6], jnt_mtx3[7], jnt_mtx3[8]])
            jnt_data.append(jnt_data_add)

        



    #rig_header = ["No.", "rigName", "dimension", "translateX", "translateX_value", "translateY", "translateY_value", "translateZ", "translateZ_value",
    #                "rotateX", "rotateX_value", "rotateY", "rotateY_value", "rotateZ", "rotateZ_value",
    #                "scaleX", "scaleX_value", "scaleY", "scaleY_value", "scaleZ", "scaleZ_value"]
    #rig_header = ["No.", "rigName", "dimension", "translateX", "translateX_value", "translateY", "translateY_value", "translateZ", "translateZ_value", "rotateX", "rotateX_value", "rotateY", "rotateY_value", "rotateZ", "rotateZ_value"]
    #rig_header = ["No.", "rigName", "dimension", "rotateX", "rotateX_value", "rotateY", "rotateY_value", "rotateZ", "rotateZ_value"]
    rig_header = ["No.", "rigName", "dimension", "translateX", "translateX_value", "translateY", "translateY_value", "translateZ", "translateZ_value",
                                        "rotate_00", "rotate_00_value", "rotate_01", "rotate_01_value", "rotate_02", "rotate_02_value",
                                        "rotate_10", "rotate_10_value", "rotate_11", "rotate_11_value", "rotate_12", "rotate_12_value",
                                        "rotate_20", "rotate_20_value", "rotate_21", "rotate_21_value", "rotate_22", "rotate_22_value"]

    #jnt_header = ["No.", "jointName", "translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "scaleX", "scaleY", "scaleZ"]
    #jnt_header = ["No.", "jointName", "translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ"]
    #jnt_header = ["No.", "jointName", "rotateX", "rotateY", "rotateZ"]
    jnt_header = ["No.", "jointName", "translateX", "translateY", "translateZ",
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