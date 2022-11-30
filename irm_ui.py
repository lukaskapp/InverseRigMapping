from PySide2 import QtCore, QtWidgets, QtGui
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui
from functools import partial
from imp import reload



def maya_main_window():
    """
    Return the Maya main window widget as a Python object
    """
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QtWidgets.QWidget)



class IRM_UI(QtWidgets.QDialog):

    dlg_instance = None

    @classmethod
    def show_dialog(cls):
        if not cls.dlg_instance:
            cls.dlg_instance = IRM_UI()

        if cls.dlg_instance.isHidden():
            cls.dlg_instance.show()
        else:
            cls.dlg_instance.raise_()
            cls.dlg_instance.activateWindow()


    def __init__(self, parent=maya_main_window()):
        super(IRM_UI, self).__init__(parent)

        self.setWindowTitle("Inverse Rig Mapping")
        self.setMinimumSize(250, 150)

        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint)

        self.create_widgets()
        self.create_layouts()
        self.create_connections()

    def create_widgets(self):
        self.prep_train_data_btn = QtWidgets.QPushButton("Prep Train Data")
        self.train_model_btn = QtWidgets.QPushButton("Train Model")
        self.prep_anim_data_btn = QtWidgets.QPushButton("Prep Anim Data")
        self.map_predict_btn = QtWidgets.QPushButton("Map Prediction")



    def create_layouts(self):      
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(3,3,3,3)
        main_layout.addWidget(self.prep_train_data_btn)
        #main_layout.addWidget(self.train_model_btn)
        main_layout.addWidget(self.prep_anim_data_btn)
        main_layout.addWidget(self.map_predict_btn)

    def create_connections(self):
        self.prep_train_data_btn.clicked.connect(self.prep_train_data)
        self.train_model_btn.clicked.connect(self.train_model)
        self.prep_anim_data_btn.clicked.connect(self.prep_anim_data)
        self.map_predict_btn.clicked.connect(self.map_predict)


    def prep_train_data(self):
        import prep_training_data
        reload(prep_training_data)
        prep_training_data.prep_data()


    def train_model(self):
        import train_model
        reload(train_model)
        train_model.train_model()

    def prep_anim_data(self):
        import prep_anim_data
        reload(prep_anim_data)
        prep_anim_data.prep_data()


    def map_predict(self):
        import apply_prediction
        reload(apply_prediction)
        apply_prediction.map_data()

try:
    irm_dialog.close()
    irm_dialog.deleteLater()
except:
    pass

irm_dialog = IRM_UI()
irm_dialog.show()
