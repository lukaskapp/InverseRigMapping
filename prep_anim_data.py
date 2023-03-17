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


    jnt_list = [jnt for jnt in cmds.ls(sl=1, typ="joint") if "_bind" in jnt and not "_end_bind" in jnt]
    jnt_list.sort()
    print("JNT LIST: ", jnt_list)


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


    header = ["No.", "jointName",
                                "rotMtx_00", "rotMtx_01", "rotMtx_02",
                                "rotMtx_10", "rotMtx_11", "rotMtx_12",
                                "rotMtx_20", "rotMtx_21", "rotMtx_22"]


    print("DATA: ", anim_data)
    file_name = "irm_anim_data.csv"
    fullpath = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data", file_name)


    with open(fullpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(anim_data)

    return fullpath.as_posix(), frames


if __name__ == "__main__":
    prep_data()