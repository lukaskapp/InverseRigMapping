"""
-----------------------------------------------------------------------------
This file has been developed within the scope of the
Technical Director course at Filmakademie Baden-Wuerttemberg.
http://technicaldirector.de

Written by Lukas Kapp
Copyright (c) 2023 Animationsinstitut of Filmakademie Baden-Wuerttemberg
-----------------------------------------------------------------------------
"""

from PySide2 import QtCore, QtWidgets, QtGui
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui
import maya.OpenMaya as om
import maya.cmds as cmds
from functools import partial
import os
from imp import reload

import utils.maya as mUtils
reload(mUtils)


def maya_main_window():
    """
    Return the Maya main window widget as a Python object
    """
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QtWidgets.QWidget)


class EditableItemDelegate(QtWidgets.QItemDelegate):
    def setModelData(self, editor, model, index):
        text = editor.text().strip()  # Remove leading/trailing spaces
        if not text or text == index.data():  # Do not update if text is empty or same as before
            return
        super().setModelData(editor, model, index)


class PlaceholderTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, parent=None, label_msg="No items added"):
        super(PlaceholderTreeWidget, self).__init__(parent)
        self.emptyLabel = QtWidgets.QLabel(label_msg, self)
        self.emptyLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.emptyLabel.hide()

    def resizeEvent(self, event):
        super(PlaceholderTreeWidget, self).resizeEvent(event)
        header_height = self.header().height()
        self.emptyLabel.setGeometry(0, header_height, self.width(), self.height() - header_height)

    def checkIfEmpty(self):
        if self.topLevelItemCount() == 0:
            self.emptyLabel.show()
        else:
            self.emptyLabel.hide()


class UnmovableSplitterHandle(QtWidgets.QSplitterHandle):
    def __init__(self, orientation, parent):
        super(UnmovableSplitterHandle, self).__init__(orientation, parent)

    def mouseMoveEvent(self, event):
        pass

class UnmovableSplitter(QtWidgets.QSplitter):
    def __init__(self, orientation, parent=None):
        super(UnmovableSplitter, self).__init__(orientation, parent)

    def createHandle(self):
        return UnmovableSplitterHandle(self.orientation(), self)



### FUNCTIONS ###


def add_selection(treeWidget, jnt_mode):
    if jnt_mode:
        sel = cmds.ls(sl=1, typ="joint")
    else:
        raw_sel = cmds.ls(sl=1, transforms=True)
        sel = []
        for transform in raw_sel:
            child_nodes = cmds.listRelatives(transform, children=True, fullPath=True) or []
            has_desired_types = any(cmds.nodeType(child) in ['nurbsCurve', 'mesh', 'nurbsSurface'] for child in child_nodes)
            if has_desired_types:
                sel.append(transform)
    for obj in sel:
        add_tree_item(treeWidget=treeWidget, name=obj, jnt_mode=jnt_mode)


def add_tree_item(treeWidget, name, jnt_mode=False):
    root = treeWidget.invisibleRootItem()
    for i in range(root.childCount()):
        if root.child(i).text(0) == name:
            om.MGlobal.displayWarning(f"Item '{name}' already exists. Skipping...")
            return

    if jnt_mode:
        attr_list = [attr for attr in mUtils.get_all_attributes(name, unlocked=False) if mUtils.check_source_connection(name, attr)]
    else:
        attr_list = mUtils.get_all_attributes(name)

    if attr_list:
        parent = QtWidgets.QTreeWidgetItem(treeWidget)
        parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsEditable)
        parent.setText(0, name)
        font = parent.font(0)
        font.setBold(True)
        parent.setFont(0, font)


        for attr in attr_list:
            child = QtWidgets.QTreeWidgetItem(parent)
            parent.addChild(child)
            #child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
            child.setText(0, attr)
            if treeWidget.columnCount() > 1:
                child.setText(1, "-50.000")
                child.setText(2, "50.000")
        
        treeWidget.expandItem(parent)


def saveFileDialog(widget, lineEdit, dialog_header, file_types):
    start_dir = lineEdit.text()
    options = QtWidgets.QFileDialog.Options()
    fileName, _ = QtWidgets.QFileDialog.getSaveFileName(widget, dialog_header, start_dir, f"{file_types.upper()} Files (*.{file_types})", options=options)
    if fileName:
        if not fileName.endswith(f".{file_types}"):
            fileName += f".{file_types}"
        lineEdit.setText(fileName)


def openFileDialog(widget, lineEdit, dialog_header, file_types):
    start_dir = lineEdit.text()
    options = QtWidgets.QFileDialog.Options()
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(widget, dialog_header, start_dir, f"{file_types.upper()} Files (*.{file_types})", options=options)
    if fileName:
        if not fileName.endswith(f".{file_types}"):
            fileName += f".{file_types}"
        lineEdit.setText(fileName)


def update_param_label(treeWidget, label):
    count = 0
    for i in range(0, treeWidget.topLevelItemCount()):
        parent = treeWidget.topLevelItem(i)
        count += parent.childCount()

    label.setText("{}({})".format(label.text().rpartition("(")[0], count))


def clear_tree(treeWidget, label):
    treeWidget.clear()
    treeWidget.checkIfEmpty()
    update_param_label(treeWidget, label)


def show_context_menu(widget, pos, treeWidget):
    menu = QtWidgets.QMenu(widget)
    delete_action = menu.addAction("Delete")
    delete_action.triggered.connect(partial(delete_items, treeWidget))
    menu.exec_(treeWidget.viewport().mapToGlobal(pos))


def delete_items(treeWidget):
    selected_items = treeWidget.selectedItems()
    for item in selected_items:
        (item.parent() or treeWidget.invisibleRootItem()).removeChild(item)


def get_treeItems_as_dict(treeWidget):
    item_dict = {}
    root = treeWidget.invisibleRootItem()
    for i in range(root.childCount()):
        parent_item = root.child(i)
        parent_name = parent_item.text(0)
        item_dict[parent_name] = []
        get_treeChildren_as_list(parent_item, item_dict[parent_name])

    return item_dict


def get_treeChildren_as_list(parent, children_list):
    for i in range(parent.childCount()):
        child_item = parent.child(i)
        child_name = child_item.text(0)

        num_columns = child_item.columnCount()
        if num_columns == 3:
            child_min_range = child_item.text(1)
            child_max_range = child_item.text(2)
            children_list.append([child_name, child_min_range, child_max_range])
        elif num_columns == 1:
            children_list.append([child_name])

        get_treeChildren_as_list(child_item, children_list)


def is_valid_dir(path):
    dir_path = os.path.dirname(path)
    return os.path.isdir(dir_path)

def is_valid_file(path):
    return os.path.isfile(path)


def check_dir_path(path):
    if not is_valid_dir(path):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Invalid file path: " + path)
        msg.setWindowTitle("File Error")
        msg.exec_()
        return False
    return True


def check_train_data(data, data_type):
    parameters = [param for param in data.values() if param]
    if not parameters:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(f"No {data_type} parameters found!")
        msg.setWindowTitle("Data Error")
        msg.exec_()
        return False
    return True  


def check_file_path(path):
    if not is_valid_file(path):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Invalid file path: " + path)
        msg.setWindowTitle("File Error")
        msg.exec_()
        return False
    return True
