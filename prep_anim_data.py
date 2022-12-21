import maya.cmds as cmds
import pymel.core as pm
import os
import csv
import maya.api.OpenMaya as om
import pathlib



def prep_data():
    sel_jnt = cmds.ls(sl=1)[0]

    if len(sel_jnt) == 0:
        om.MGlobal.displayError("Nothing selected! Please select one or more joints!")
        return

    if cmds.listRelatives(sel_jnt, ad=1, typ="joint"):
        jnt_list = [obj for obj in cmds.listRelatives(sel_jnt, ad=1, typ="joint") if "_bind" in obj and not "_end_" in obj]
        jnt_list.append(sel_jnt)
        jnt_list.reverse()
    else:
        jnt_list = [sel_jnt]

    anim_data = []
    current_frame = cmds.currentTime(q=1)
    frames = []
    for jnt in jnt_list:
        frames.extend(list(set(cmds.keyframe(jnt, q=1))))    
    frames = list(set(frames))

    for i, jnt in enumerate(jnt_list):
        for frame in frames:
            cmds.currentTime(frame)
            jnt_mtx = pm.dt.TransformationMatrix(cmds.xform(jnt, m=1, q=1, os=1))
            jnt_rot_mtx3 = [x for mtx in jnt_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]
            jnt_trans = jnt_mtx.getTranslation("object")

            jnt_data = [i, jnt]
            #jnt_data.extend(jnt_trans)
            jnt_data.extend(jnt_rot_mtx3)
            #jnt_data.extend(jnt_scale)
            anim_data.append(jnt_data)
    
    cmds.currentTime(current_frame)
    print("FRAMES: ", frames)
    #header = ["No.", "jointName", "translateX", "translateY", "translateZ",
    #                                "rotate_00", "rotate_01", "rotate_02",
    #                                "rotate_10", "rotate_11", "rotate_12",
    #                                "rotate_20", "rotate_21", "rotate_22"]

    header = ["No.", "jointName",
                                    "rotate_00", "rotate_01", "rotate_02",
                                    "rotate_10", "rotate_11", "rotate_12",
                                    "rotate_20", "rotate_21", "rotate_22"]


    print("DATA: ", anim_data)
    file_name = "anim_data_01.csv"
    fullpath = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", file_name)


    with open(fullpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(anim_data)

    return fullpath.as_posix(), frames



if __name__ == "__main__":
    prep_data()