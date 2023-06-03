from PySide2 import QtCore, QtWidgets, QtGui
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui
import maya.cmds as cmds
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


class CustomTreeWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, data):
        super(CustomTreeWidgetItem, self).__init__(data)

    def flags(self, column):
        if column == 0:  # First column
            return super().flags(column) | QtCore.Qt.ItemIsEditable  # Not editable
        else:  # Second and third columns
            return super().flags(column) | QtCore.Qt.ItemIsEditable  # Editable



### FUNCTIONS ###

def add_selection(treeWidget):
    sel = cmds.ls(sl=1)
    for obj in sel:
        add_tree_item(treeWidget=treeWidget, item_list=[obj])


def add_tree_item(treeWidget, item_list):
    for i, name in enumerate(item_list):
        parent = QtWidgets.QTreeWidgetItem(treeWidget)
        #parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsEditable)
        parent.setText(0, name)
        font = parent.font(0)
        font.setBold(True)
        parent.setFont(0, font)
        
        for attr in mUtils.get_all_attributes(name):
            child = CustomTreeWidgetItem([attr, "-50", "50"])
            parent.addChild(child)

            #child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
            #child.setText(0, attr)
            #child.setText(1, "-50.00")
            #child.setText(2, "50.00")
