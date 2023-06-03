from PySide2 import QtCore, QtWidgets, QtGui
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui


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




