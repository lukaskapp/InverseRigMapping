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

def filter_attrs_from_dict(dict_entry):
    return [attr[0] for attr in dict_entry]


def restore_defaults(ctrl):
    '''
    set all attributes on obj to default values
    '''
    for attr in get_all_attributes(ctrl):
        cmds.setAttr("{}.{}".format(ctrl, attr), cmds.attributeQuery(attr, node=ctrl, ld=1)[0])


def check_source_connection(obj, attr):
    '''
    check given attribute for incoming connections
    '''
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
    

