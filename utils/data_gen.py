


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
