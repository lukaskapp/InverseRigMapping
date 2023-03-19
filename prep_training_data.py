import maya.cmds as cmds
import pymel.core as pm
import maya.api.OpenMaya as om
import math
import os
import random
import csv
import pathlib


def get_all_attributes(obj, unlocked=True):
    '''
    return all attributes of obj if attribute is unlocked, keyable, scalar and not of type enum or bool
    '''
    return [attr for attr in cmds.listAttr(obj, unlocked=unlocked, keyable=True, scalar=True) if not cmds.attributeQuery(attr, node=obj, at=1) in ["enum", "bool"]]

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


def check_transformLimit(ctrl, axis="rotateX", default_min=-180, default_max=180):
    limit_dict = {"translateX":[1,0,0, 0,0,0, 0,0,0], "translateY":[0,1,0, 0,0,0, 0,0,0], "translateZ":[0,0,1, 0,0,0, 0,0,0],
                    "rotateX":[0,0,0, 1,0,0, 0,0,0], "rotateY":[0,0,0, 0,1,0, 0,0,0], "rotateZ":[0,0,0, 0,0,1, 0,0,0],
                    "scaleX":[0,0,0, 0,0,0, 1,0,0], "scaleY":[0,0,0, 0,0,0, 0,1,0], "scaleZ":[0,0,0, 0,0,0, 0,0,1]}

    if cmds.transformLimits(ctrl, q=1, etx=limit_dict[axis][0], ety=limit_dict[axis][1], etz=limit_dict[axis][2],
                                        erx=limit_dict[axis][3], ery=limit_dict[axis][4], erz=limit_dict[axis][5],
                                        esx=limit_dict[axis][6], esy=limit_dict[axis][7], esz=limit_dict[axis][8])[0]:
        limit_min =  cmds.transformLimits(ctrl, q=1, tx=limit_dict[axis][0], ty=limit_dict[axis][1], tz=limit_dict[axis][2],
                                                    rx=limit_dict[axis][3], ry=limit_dict[axis][4], rz=limit_dict[axis][5],
                                                    sx=limit_dict[axis][6], sy=limit_dict[axis][7], sz=limit_dict[axis][8])[0]
    else:
        limit_min = default_min

    if cmds.transformLimits(ctrl, q=1, etx=limit_dict[axis][0], ety=limit_dict[axis][1], etz=limit_dict[axis][2],
                                        erx=limit_dict[axis][3], ery=limit_dict[axis][4], erz=limit_dict[axis][5],
                                        esx=limit_dict[axis][6], esy=limit_dict[axis][7], esz=limit_dict[axis][8])[1]:
        limit_max =  cmds.transformLimits(ctrl, q=1, tx=limit_dict[axis][0], ty=limit_dict[axis][1], tz=limit_dict[axis][2],
                                                    rx=limit_dict[axis][3], ry=limit_dict[axis][4], rz=limit_dict[axis][5],
                                                    sx=limit_dict[axis][6], sy=limit_dict[axis][7], sz=limit_dict[axis][8])[1]
    else:
        limit_max = default_max

    return limit_min, limit_max

def query_visibility(obj): # check obj parents for vis flag
    if cmds.getAttr("{}.v".format(obj)):
        while cmds.listRelatives(obj, p=1):
            parent = cmds.listRelatives(obj, p=1)[0]
            if cmds.getAttr("{}.v".format(parent)):
                obj = parent
            else:
                return False
        return True
    else:
        return False
    

def build_header(base_header=["No.", "rigName", "dimension"], attr_list=[]):

    for attr in attr_list:
        if attr in ["translateX", "translateY", "translateZ"]:
            base_header.append(attr)

    if "rot_mtx" in attr_list:
        base_header.extend(["rotMtx_00", "rotMtx_01", "rotMtx_02",
                            "rotMtx_10", "rotMtx_11", "rotMtx_12",
                            "rotMtx_20", "rotMtx_21", "rotMtx_22"])

    for attr in attr_list:
        if attr in ["scaleX", "scaleY", "scaleZ"]:
            base_header.append(attr) 

    base_header.extend([attr for attr in attr_list if not attr in ["translateX", "translateY", "translateZ", "rot_mtx", "scaleX", "scaleY", "scaleZ"]])

    return base_header  


def get_attr_dimension(attr_list, rotation=True):
    # set dimension to length of attr_list; if rotate in list, remove rotate and add matrix3 (9 dimension)
    if rotation:
        return len([attr for attr in attr_list if not attr in ["rotateX", "rotateY", "rotateZ"]]) + 9
    else:
        return len(attr_list)

def check_for_rotation(attr_list):
    if ("rotateX" or "rotateY" or "rotateZ") in attr_list:
        return True
    else:
        return False



##############################################################################################################################



def prep_data(rig_input_data, jnt_input_data):
    if len(rig_input_data) == 0 or len(rig_input_data) == 0:
        om.MGlobal.displayError("Nothing selected! Please select one control and one or more joints!")
        return

    # filter selection into joints and controls
    #ctrl_list = [ctrl for ctrl in cmds.ls(sl=1, typ="transform") if "_ctrl" in ctrl and not "_srtBuffer" in ctrl and query_visibility(ctrl)]
    ctrl_list = [ctrl for ctrl in rig_input_data if query_visibility(ctrl)]
    ctrl_list.sort()
    #print("CTRL LIST: ", ctrl_list)

    #jnt_list = [jnt for jnt in cmds.ls(sl=1, typ="joint") if "_bind" in jnt and not "_end_bind" in jnt]
    jnt_list = [jnt for jnt in jnt_input_data]
    jnt_list.sort()
    #print("JNT LIST: ", jnt_list)


    for ctrl in ctrl_list:
        restore_defaults(ctrl)


    ctrl_unique_attrs = list(set([unique_attr for ctrl in ctrl_list for unique_attr in get_all_attributes(ctrl)]))
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
        for attr in get_all_attributes(jnt, unlocked=False):
            if check_source_connection(jnt, attr):
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
    rig_header = build_header(base_header=["No.", "rigName", "dimension"], attr_list=ctrl_unique_attrs)
    jnt_header = build_header(base_header=["No.", "jointName", "dimension"], attr_list=jnt_unique_attrs)


    #print("")
    #print("-----------------------------------------------------------")
    #print("")

    rig_data = []
    jnt_data = []

    iterations = 1000
    for i in range(iterations):
        print("Progress: {}/{}".format(str(i+1).zfill(4), iterations))
        for ctrl_index, ctrl in enumerate(ctrl_list):
            # only get integer and float attributes of selected control
            attr_list = get_all_attributes(ctrl)
            #print("ATTR LIST: ", attr_list)
            
            
            # check if rotation is in attr list
            rotation = check_for_rotation(attr_list)
            #print("ROT: ", rotation)

            # set dimension to length of attr_list; if rotate in list, remove rotate and add matrix3 (9 dimension)
            attr_dimension = get_attr_dimension(attr_list, rotation)

            
            #print("DIMENSION: ", attr_dimension)
            #print("-----------------------------------------------------------")


            default_rand_min = -50
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
                        rand_max = -default_rand_min

                elif attr in ["rotateX", "rotateY", "rotateZ"]: # if rotation, translate or scale is used, check limits and use those for min/max range
                    rand_min, rand_max = check_transformLimit(ctrl, axis=attr, default_min=-180, default_max=180)
                elif attr in ["translateX", "translateY", "translateZ"]:
                    rand_min, rand_max = check_transformLimit(ctrl, axis=attr, default_min=default_rand_min, default_max=-default_rand_min)
                elif attr in ["scaleX", "scaleY", "scaleZ"]:
                    rand_min, rand_max = check_transformLimit(ctrl, axis=attr, default_min=0.01, default_max=10)

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
            attr_list = [attr for attr in get_all_attributes(jnt, unlocked=False) if check_source_connection(jnt, attr)]
            
            rotation = check_for_rotation(attr_list)
            jnt_dimension = get_attr_dimension(attr_list, rotation)
            
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
        restore_defaults(ctrl)


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