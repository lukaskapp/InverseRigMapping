import maya.cmds as cmds
import pymel.core as pm
import os
import csv
import maya.api.OpenMaya as om
import pathlib



def prep_data(anim_input_data, jnt_path):
    jnt_list = [jnt for jnt in anim_input_data.keys()]
    jnt_list.sort()

    with open(jnt_path, "r") as f:
        reader = csv.reader(f)
        anim_header = next(reader, None) # get header
        jnt_data = [row for row in reader if len(row) != 0]

        jnt_names = list(set([entry[1] for entry in jnt_data])) # get unique jnt names
        jnt_dimension_list = [entry[2] for entry in jnt_data[:len(jnt_names)]]
                    

    anim_data = []
    current_frame = cmds.currentTime(q=1)
    frames = []
    for jnt in jnt_list:
        frames.extend(list(set(cmds.keyframe(jnt, q=1))))
    frames = list(set(frames))

    for frame_index, frame in enumerate(frames):
        for i, jnt in enumerate(jnt_list):
            cmds.currentTime(frame)
            frame_data = [i, jnt, jnt_dimension_list[i]]
            frame_data.extend(["n/a" for i in range(len(anim_header)-3)])
            
            train_values = jnt_data[i][3:]
            attr_list = [anim_header[train_values.index(value) + 3] for value in train_values if value != "n/a"]

            for attr in attr_list:
                if not "rotMtx_" in attr:
                    # replace only used attr of ctrls in n/a list, rest stays at n/a
                    frame_data[anim_header.index(attr)] = cmds.getAttr("{}.{}".format(jnt, attr))

            rotation = [attr for attr in attr_list if "rotMtx_" in attr]
            if rotation:
                jnt_mtx = pm.dt.TransformationMatrix(cmds.xform(jnt, m=1, q=1, os=1))
                jnt_rot_mtx3 = [x for mtx in jnt_mtx.asRotateMatrix()[:-1] for x in mtx[:-1]]

                start_index = anim_header.index("rotMtx_00") # get index of first rotMtx entry in anim_header and start replacing rotMtx values from there
                for mtx_index, rot_mtx in enumerate(jnt_rot_mtx3):
                    frame_data[start_index + mtx_index] = rot_mtx

            anim_data.append(frame_data)
        cmds.progressWindow(edit=True, progress=(frame_index/len(frames))*100, status=('Getting Animation Data...'))

    
    cmds.currentTime(current_frame)
    
    # save anim data as json
    anim_path = pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "anim_data/irm_anim_data.csv")
    with open(anim_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(anim_header)
        writer.writerows(anim_data)

    return anim_path.as_posix(), frames

