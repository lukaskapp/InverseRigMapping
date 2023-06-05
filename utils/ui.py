from PySide2 import QtCore, QtWidgets, QtGui
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui
import maya.OpenMaya as om
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

def add_selection(treeWidget):
    sel = cmds.ls(sl=1)
    for obj in sel:
        add_tree_item(treeWidget=treeWidget, name=obj)


def add_tree_item(treeWidget, name):
    root = treeWidget.invisibleRootItem()
    for i in range(root.childCount()):
        if root.child(i).text(0) == name:
            om.MGlobal.displayWarning(f"Item '{name}' already exists. Skipping...")
            return

    parent = QtWidgets.QTreeWidgetItem(treeWidget)
    parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsEditable)
    parent.setText(0, name)
    font = parent.font(0)
    font.setBold(True)
    parent.setFont(0, font)
    
    for attr in mUtils.get_all_attributes(name):
        child = QtWidgets.QTreeWidgetItem(parent)
        parent.addChild(child)
        #child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
        child.setText(0, attr)
        child.setText(1, "-50.000")
        child.setText(2, "50.000")
    
    treeWidget.expandItem(parent)



