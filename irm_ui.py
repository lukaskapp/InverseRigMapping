from PySide2 import QtCore, QtWidgets, QtGui
from functools import partial
import json
import pathlib
import os
from imp import reload

import utils.ui as uiUtils
reload(uiUtils)



class DataGenWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DataGenWidget, self).__init__(parent)

        self.create_widgets()
        self.create_layouts()
        self.create_connections()

        self.rig_tree.checkIfEmpty()
        self.jnt_tree.checkIfEmpty()

    def create_widgets(self):
        # rig widgets   
        self.rig_param_label = QtWidgets.QLabel("Control Rig Parameters (0)")
        self.rig_param_label.setStyleSheet("color: #00ff6e;")
        self.rig_clear_btn = QtWidgets.QPushButton("Clear All")
        self.rig_add_btn = QtWidgets.QPushButton("Add")
            
        # rig tree view widget
        rig_msg = 'Add rig control parameters using the "Add" button.\nMake sure you have at least one object selected.\nAdjust the parameter range and delete unwanted parameters by right-clicking and choosing "Delete".'
        self.rig_tree = uiUtils.PlaceholderTreeWidget(self, rig_msg)
        self.rig_tree.setHeaderLabels(['Control Name', 'Min', 'Max'])
        self.rig_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.rig_tree.setItemDelegate(uiUtils.EditableItemDelegate(self.rig_tree))
        self.rig_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        rig_header = self.rig_tree.header()
        rig_header.setStretchLastSection(False)
        rig_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        rig_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        rig_header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        rig_header.resizeSection(1, 100)
        rig_header.resizeSection(2, 100)


        # joint widgets
        self.jnt_param_label = QtWidgets.QLabel("Joint Parameters (0)")
        self.jnt_param_label.setStyleSheet("color: #00ff6e;")
        self.jnt_clear_btn = QtWidgets.QPushButton("Clear All")
        self.jnt_add_btn = QtWidgets.QPushButton("Add")

        # joint tree view widget
        jnt_msg = 'Add joints using the "Add" button.\nMake sure you have at least one joint selected.\nOnly connected parameters are added. Delete unwanted ones by right-clicking and choosing "Delete".'
        self.jnt_tree = uiUtils.PlaceholderTreeWidget(self, jnt_msg)
        self.jnt_tree.setHeaderLabels(['Joint Name'])
        self.jnt_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.jnt_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        jnt_header = self.jnt_tree.header()
        jnt_header.setStretchLastSection(False)
        jnt_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)


        # train data settings
        self.setting_widget = QtWidgets.QWidget()
        self.paramProperties_label = QtWidgets.QLabel("Parameter Properties:")

        self.minRange_label = QtWidgets.QLabel("Minimum:")
        self.minRange_line = QtWidgets.QLineEdit("-50.000")
        self.minRange_line.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("^-?\d+(\.\d{0,3})?$")))
        self.minRange_line.setEnabled(False)

        self.maxRange_label = QtWidgets.QLabel("Maximum:")
        self.maxRange_line = QtWidgets.QLineEdit("50.000")
        self.maxRange_line.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("^-?\d+(\.\d{0,3})?$")))
        self.maxRange_line.setEnabled(False)

        self.dataSpacer01_label = QtWidgets.QLabel()
        self.dataSpacer01_label.setFixedHeight(10)

        self.numPoses_label = QtWidgets.QLabel("Number of Poses:")
        self.numPoses_line = QtWidgets.QLineEdit("1000")
        self.numPoses_line.setValidator(QtGui.QIntValidator(1, 9999999))

        self.dataSpacer02_label = QtWidgets.QLabel()
        self.dataSpacer02_label.setFixedHeight(50)


        # train model settings
        self.lr_label = QtWidgets.QLabel("Learning Rate:")
        self.lr_line = QtWidgets.QLineEdit("0.01")
        self.lr_line.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("^\d+(\.\d{0,8})?$")))

        self.epoch_label = QtWidgets.QLabel("Epochs:")
        self.epoch_line = QtWidgets.QLineEdit("100")
        self.epoch_line.setValidator(QtGui.QIntValidator(1, 9999999))

        self.forceCPU_label = QtWidgets.QLabel("Force CPU:")
        self.forceCPU_checkBox = QtWidgets.QCheckBox()

        self.trainSpacer01_label = QtWidgets.QLabel()
        self.trainSpacer01_label.setFixedHeight(50)


        # output settings
        self.outRig_label = QtWidgets.QLabel("Control Rig File:")
        self.outRig_line = QtWidgets.QLineEdit()
        self.outRig_line.setText(str(pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig/irm_rig_data.csv")))
        self.outRig_line.setReadOnly(True)
        self.outRig_btn = QtWidgets.QPushButton("<")
        self.outRig_btn.setFixedSize(20,20)

        self.outJnt_label = QtWidgets.QLabel("Joint File:")
        self.outJnt_line = QtWidgets.QLineEdit()
        self.outJnt_line.setText(str(pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt/irm_jnt_data.csv")))
        self.outJnt_line.setReadOnly(True)
        self.outJnt_btn = QtWidgets.QPushButton("<")
        self.outJnt_btn.setFixedSize(20,20)

        self.outSpacer01_label = QtWidgets.QLabel()
        self.outSpacer01_label.setFixedHeight(10)

        self.outModel_label = QtWidgets.QLabel("Model File:")
        self.outModel_line = QtWidgets.QLineEdit()
        self.outModel_line.setText(str(pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "trained_model/trained_model.pt")))
        self.outModel_line.setReadOnly(True)
        self.outModel_btn = QtWidgets.QPushButton("<")
        self.outModel_btn.setFixedSize(20,20)

        self.outSpacer02_label = QtWidgets.QLabel()
        self.outSpacer02_label.setFixedHeight(50)


        # buttons
        self.generate_btn = QtWidgets.QPushButton("Generate Train Data")
        self.train_btn = QtWidgets.QPushButton("Train Model")


    def create_layouts(self): 
        # main UI layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(3,3,3,3)

        # left UI side - rig and joint tree views
        tree_widget = QtWidgets.QWidget()
        tree_layout = QtWidgets.QVBoxLayout()
        tree_widget.setLayout(tree_layout)

        # rig layout
        rig_group = QtWidgets.QGroupBox()
        tree_layout.addWidget(rig_group)
        rig_attr_layout = QtWidgets.QVBoxLayout()
        rig_group.setLayout(rig_attr_layout)

        rig_btn_layout = QtWidgets.QHBoxLayout()
        rig_btn_layout.addWidget(self.rig_param_label)
        rig_btn_layout.addWidget(self.rig_clear_btn)
        rig_btn_layout.addWidget(self.rig_add_btn)
        rig_attr_layout.addLayout(rig_btn_layout)

        rig_attr_layout.addWidget(self.rig_tree)

        # jnt layout
        jnt_group = QtWidgets.QGroupBox()
        tree_layout.addWidget(jnt_group)
        jnt_attr_layout = QtWidgets.QVBoxLayout()
        jnt_group.setLayout(jnt_attr_layout)

        jnt_btn_layout = QtWidgets.QHBoxLayout()
        jnt_btn_layout.addWidget(self.jnt_param_label)        
        jnt_btn_layout.addWidget(self.jnt_clear_btn)        
        jnt_btn_layout.addWidget(self.jnt_add_btn) 
        jnt_attr_layout.addLayout(jnt_btn_layout)
        
        jnt_attr_layout.addWidget(self.jnt_tree)


        # right UI side - settings for data gen and model training
        settings_widget = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout()
        settings_widget.setLayout(settings_layout)

        # data gen settings
        data_widget = QtWidgets.QGroupBox("Generator Settings")
        data_widget.setObjectName("data_box")
        data_widget.setStyleSheet("QGroupBox#data_box { color: #00ff6e; }")
        data_layout = QtWidgets.QGridLayout()
        data_widget.setLayout(data_layout)

        dataAlign_widget = QtWidgets.QWidget()
        dataAlign_widget.setFixedWidth(110)
        dataAlign_widget.setFixedHeight(1)
        data_layout.addWidget(dataAlign_widget, 0, 0)

        data_layout.addWidget(self.paramProperties_label, 1, 0)

        data_layout.addWidget(self.minRange_label, 2, 0)
        data_layout.addWidget(self.minRange_line, 2, 1)

        data_layout.addWidget(self.maxRange_label, 3, 0)
        data_layout.addWidget(self.maxRange_line, 3, 1)

        data_layout.addWidget(self.dataSpacer01_label, 4, 0)

        data_layout.addWidget(self.numPoses_label, 5, 0)
        data_layout.addWidget(self.numPoses_line, 5, 1)

        data_layout.addWidget(self.dataSpacer02_label, 6, 0)


        # train settings
        train_widget = QtWidgets.QGroupBox("Train Settings")
        train_widget.setStyleSheet("QGroupBox { color: #00ff6e; }")
        train_layout = QtWidgets.QGridLayout()
        train_widget.setLayout(train_layout)

        trainAlign_widget = QtWidgets.QWidget()
        trainAlign_widget.setFixedWidth(110)
        trainAlign_widget.setFixedHeight(1)
        train_layout.addWidget(trainAlign_widget, 0, 0)

        train_layout.addWidget(self.lr_label, 1, 0)
        train_layout.addWidget(self.lr_line, 1, 1)

        train_layout.addWidget(self.epoch_label, 2, 0)
        train_layout.addWidget(self.epoch_line, 2, 1)

        train_layout.addWidget(self.forceCPU_label, 3, 0)
        train_layout.addWidget(self.forceCPU_checkBox, 3, 1)
        
        train_layout.addWidget(self.trainSpacer01_label, 4, 0)


        # output settings
        output_widget = QtWidgets.QGroupBox("Output Settings")
        output_widget.setStyleSheet("QGroupBox { color: #00ff6e; }")
        output_layout = QtWidgets.QGridLayout()
        output_widget.setLayout(output_layout)

        outAlign_widget = QtWidgets.QWidget()
        outAlign_widget.setFixedWidth(110)
        outAlign_widget.setFixedHeight(1)
        output_layout.addWidget(outAlign_widget, 0, 0)

        output_layout.addWidget(self.outRig_label, 1, 0)
        output_layout.addWidget(self.outRig_line, 1, 1)
        output_layout.addWidget(self.outRig_btn, 1, 2)

        output_layout.addWidget(self.outJnt_label, 2, 0)
        output_layout.addWidget(self.outJnt_line, 2, 1)
        output_layout.addWidget(self.outJnt_btn, 2, 2)

        output_layout.addWidget(self.outSpacer01_label, 3, 0)

        output_layout.addWidget(self.outModel_label, 4, 0)
        output_layout.addWidget(self.outModel_line, 4, 1)
        output_layout.addWidget(self.outModel_btn, 4, 2)

        output_layout.addWidget(self.outSpacer02_label, 5, 0)


        # data/train settings spliiter
        settings_splitter = uiUtils.UnmovableSplitter(QtCore.Qt.Vertical)
        settings_splitter.addWidget(data_widget)
        settings_splitter.addWidget(train_widget)
        settings_splitter.addWidget(output_widget)
        settings_splitter.setSizes([5000, 5000, 5000])
        settings_layout.addWidget(settings_splitter)
        
        # button layout
        outBtn_layout = QtWidgets.QHBoxLayout()
        outBtn_layout.addWidget(self.generate_btn)
        outBtn_layout.addWidget(self.train_btn)
        settings_layout.addLayout(outBtn_layout)


        # main layout splitter
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(tree_widget)
        main_splitter.addWidget(settings_widget)
        main_splitter.setSizes([7000, 3000])
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        

    def create_connections(self):
        # rig UI
        self.rig_clear_btn.clicked.connect(partial(uiUtils.clear_tree, self.rig_tree, self.rig_param_label))
        self.rig_add_btn.clicked.connect(partial(self.add_tree_item, self.rig_tree, self.rig_param_label, jnt_mode=False))
        
        self.rig_tree.customContextMenuRequested.connect(self.show_rig_context_menu)
        self.rig_tree.itemChanged.connect(self.rig_tree.checkIfEmpty)
        self.rig_tree.itemSelectionChanged.connect(self.update_param_range)
        
        # joint UI
        self.jnt_clear_btn.clicked.connect(partial(uiUtils.clear_tree, self.jnt_tree, self.jnt_param_label))
        self.jnt_add_btn.clicked.connect(partial(self.add_tree_item, self.jnt_tree, self.jnt_param_label, jnt_mode=True))
        
        self.jnt_tree.customContextMenuRequested.connect(self.show_jnt_context_menu)
        self.jnt_tree.itemChanged.connect(self.jnt_tree.checkIfEmpty)

        # data/train settings
        self.minRange_line.editingFinished.connect(self.update_min_range)
        self.maxRange_line.editingFinished.connect(self.update_max_range)

        self.outRig_btn.clicked.connect(self.set_rigData_outPath)
        self.outJnt_btn.clicked.connect(self.set_jntData_outPath)
        self.outModel_btn.clicked.connect(self.set_model_outPath)

        self.generate_btn.clicked.connect(self.generate_train_data)
        self.train_btn.clicked.connect(self.train_model)
    

    def set_rigData_outPath(self):
        uiUtils.saveFileDialog(self, self.outRig_line, "Save Control Rig Train Data", "csv")

    def set_jntData_outPath(self):
        uiUtils.saveFileDialog(self, self.outJnt_line, "Save Joint Train Data", "csv")

    def set_model_outPath(self):
        uiUtils.saveFileDialog(self, self.outModel_line, "Save Trained Model", "pt")

    def add_tree_item(self, treeWidget, label, jnt_mode):
        uiUtils.add_selection(treeWidget, jnt_mode)
        uiUtils.update_param_label(treeWidget, label)

    def show_rig_context_menu(self, pos):
        uiUtils.show_context_menu(self, pos, self.rig_tree)
        self.rig_tree.checkIfEmpty()
        uiUtils.update_param_label(self.rig_tree, self.rig_param_label)

    def show_jnt_context_menu(self, pos):
        uiUtils.show_context_menu(self, pos, self.jnt_tree)
        self.jnt_tree.checkIfEmpty()
        uiUtils.update_param_label(self.jnt_tree, self.jnt_param_label)

        
    def update_param_range(self):
        selected_items = self.rig_tree.selectedItems()

        if selected_items:
            item = selected_items[0]

            if item.childCount() > 0:
                return
            else:
                self.minRange_line.setEnabled(True)
                self.maxRange_line.setEnabled(True)
                self.minRange_line.setText(item.text(1))
                self.maxRange_line.setText(item.text(2))
        else:
            self.minRange_line.setEnabled(False)
            self.maxRange_line.setEnabled(False)


    def ensure_three_digits(self, lineWidget):
        value = float(lineWidget.text())
        formatted_value = "{:.3f}".format(value)
        lineWidget.setText(formatted_value)

    def ensure_notZero(self, lineWidget):
        value = int(lineWidget.text())
        print(value)
        if value == 0:
            lineWidget.setText("1")

    def update_min_range(self):
        self.update_selected_items(self.minRange_line, 1)
        self.ensure_three_digits(self.minRange_line)

    def update_max_range(self):
        self.update_selected_items(self.maxRange_line, 2)
        self.ensure_three_digits(self.maxRange_line)


    def update_selected_items(self, lineWidget, column):
        selected_items = self.rig_tree.selectedItems()
        
        for item in selected_items:
            if item.childCount() > 0:
                continue
            try:
                item.setText(column, "{:.3f}".format(float(lineWidget.text())))
            except ValueError:
                pass


    def generate_train_data(self):
        rig_input_data = uiUtils.get_treeItems_as_dict(treeWidget=self.rig_tree)
        jnt_input_data = uiUtils.get_treeItems_as_dict(treeWidget=self.jnt_tree)

        rig_path = self.outRig_line.text()
        jnt_path = self.outJnt_line.text()

        print("RIG INPUT DATA: ", rig_input_data)
        print("RIG OUT PATH: ", rig_path)
        print("JNT INPUT DATA: ", jnt_input_data)      
        print("JNT OUT PATH: ", jnt_path)

        uiUtils.check_file_path(path=rig_path)
        uiUtils.check_file_path(path=jnt_path)

        import generate_train_data
        reload(generate_train_data)
        #generate_train_data.generate_data(rig_input_data=rig_data, jnt_input_data=jnt_data,
        #                            rig_out=rig_path, jnt_out=jnt_path)


    def train_model(self):
        import train_model
        reload(train_model)
        train_model.train_model()



class PredictWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PredictWidget, self).__init__(parent)

        self.create_widgets()
        self.create_layouts()
        self.create_connections()

        self.anim_tree.checkIfEmpty()


    def create_widgets(self):
        # anim widgets   
        self.anim_param_label = QtWidgets.QLabel("Animation Parameters (0)")
        self.anim_param_label.setStyleSheet("color: #00ff6e;")
        self.anim_clear_btn = QtWidgets.QPushButton("Clear All")
        self.anim_add_btn = QtWidgets.QPushButton("Add")
            
        # anim tree view widget
        anim_msg = 'Add animated parameters using the "Add" button.\nMake sure you have at least one joint selected.'
        self.anim_tree = uiUtils.PlaceholderTreeWidget(self, anim_msg)
        self.anim_tree.setHeaderLabels(['Joint Name'])
        self.anim_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.anim_tree.setItemDelegate(uiUtils.EditableItemDelegate(self.anim_tree))
        self.anim_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        anim_header = self.anim_tree.header()
        anim_header.setStretchLastSection(False)
        anim_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)



        # predict settings
        self.setting_widget = QtWidgets.QWidget()

        self.outRig_label = QtWidgets.QLabel("Control Rig Data:")
        self.outRig_line = QtWidgets.QLineEdit()
        self.outRig_line.setText(str(pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/rig/irm_rig_data.csv")))
        self.outRig_line.setReadOnly(True)
        self.outRig_btn = QtWidgets.QPushButton("<")
        self.outRig_btn.setFixedSize(20,20)

        self.outJnt_label = QtWidgets.QLabel("Joint Data:")
        self.outJnt_line = QtWidgets.QLineEdit()
        self.outJnt_line.setText(str(pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "training_data/jnt/irm_jnt_data.csv")))
        self.outJnt_line.setReadOnly(True)
        self.outJnt_btn = QtWidgets.QPushButton("<")
        self.outJnt_btn.setFixedSize(20,20)

        self.outSpacer01_label = QtWidgets.QLabel()
        self.outSpacer01_label.setFixedHeight(10)

        self.outModel_label = QtWidgets.QLabel("Trained Model:")
        self.outModel_line = QtWidgets.QLineEdit()
        self.outModel_line.setText(str(pathlib.PurePath(os.path.normpath(os.path.dirname(os.path.realpath(__file__))), "trained_model/trained_model.pt")))
        self.outModel_line.setReadOnly(True)
        self.outModel_btn = QtWidgets.QPushButton("<")
        self.outModel_btn.setFixedSize(20,20)

        self.outSpacer02_label = QtWidgets.QLabel()
        self.outSpacer02_label.setFixedHeight(500)

        # buttons
        self.predict_btn = QtWidgets.QPushButton("Map Prediction")


    def create_layouts(self): 
        # main UI layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(3,3,3,3)


        # left UI side - anim parameters
        tree_widget = QtWidgets.QWidget()
        tree_layout = QtWidgets.QVBoxLayout()
        tree_widget.setLayout(tree_layout)


        # anim layout
        anim_group = QtWidgets.QGroupBox()
        tree_layout.addWidget(anim_group)
        anim_attr_layout = QtWidgets.QVBoxLayout()
        anim_group.setLayout(anim_attr_layout)

        anim_btn_layout = QtWidgets.QHBoxLayout()
        anim_btn_layout.addWidget(self.anim_param_label)
        anim_btn_layout.addWidget(self.anim_clear_btn)
        anim_btn_layout.addWidget(self.anim_add_btn)
        anim_attr_layout.addLayout(anim_btn_layout)

        anim_attr_layout.addWidget(self.anim_tree)


        # right UI side - settings for prediction
        settings_widget = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout()
        settings_widget.setLayout(settings_layout)

        predict_widget = QtWidgets.QGroupBox("Predict Settings")
        predict_widget.setStyleSheet("QGroupBox { color: #00ff6e; }")
        predict_layout = QtWidgets.QGridLayout()
        predict_widget.setLayout(predict_layout)

        predictAlign_widget = QtWidgets.QWidget()
        predictAlign_widget.setFixedWidth(110)
        predictAlign_widget.setFixedHeight(1)
        predict_layout.addWidget(predictAlign_widget, 0, 0)

        predictAlign_widget = QtWidgets.QWidget()
        predictAlign_widget.setFixedWidth(110)
        predictAlign_widget.setFixedHeight(1)
        predict_layout.addWidget(predictAlign_widget, 0, 0)

        predict_layout.addWidget(self.outRig_label, 1, 0)
        predict_layout.addWidget(self.outRig_line, 1, 1)
        predict_layout.addWidget(self.outRig_btn, 1, 2)

        predict_layout.addWidget(self.outJnt_label, 2, 0)
        predict_layout.addWidget(self.outJnt_line, 2, 1)
        predict_layout.addWidget(self.outJnt_btn, 2, 2)

        predict_layout.addWidget(self.outSpacer01_label, 3, 0)

        predict_layout.addWidget(self.outModel_label, 4, 0)
        predict_layout.addWidget(self.outModel_line, 4, 1)
        predict_layout.addWidget(self.outModel_btn, 4, 2)

        predict_layout.addWidget(self.outSpacer02_label, 5, 0)

        settings_layout.addWidget(predict_widget)
        
        # button layout
        outBtn_layout = QtWidgets.QHBoxLayout()
        outBtn_layout.addWidget(self.predict_btn)
        settings_layout.addLayout(outBtn_layout)


        # main layout splitter
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(tree_widget)
        main_splitter.addWidget(settings_widget)
        main_splitter.setSizes([7000, 3000])
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)


    def create_connections(self):
        # anim UI
        self.anim_clear_btn.clicked.connect(partial(uiUtils.clear_tree, self.anim_tree, self.anim_param_label))
        self.anim_add_btn.clicked.connect(partial(self.add_tree_item, self.anim_tree, self.anim_param_label, jnt_mode=True))
        
        self.anim_tree.customContextMenuRequested.connect(self.show_anim_context_menu)
        self.anim_tree.itemChanged.connect(self.anim_tree.checkIfEmpty)

        # predict settings
        self.outRig_btn.clicked.connect(self.set_rigData_outPath)
        self.outJnt_btn.clicked.connect(self.set_jntData_outPath)
        self.outModel_btn.clicked.connect(self.set_model_outPath)

        self.predict_btn.clicked.connect(self.map_prediction)
        

    def set_rigData_outPath(self):
        uiUtils.openFileDialog(self, self.outRig_line, "Load Control Rig Train Data", "csv")

    def set_jntData_outPath(self):
        uiUtils.openFileDialog(self, self.outJnt_line, "Load Joint Train Data", "csv")

    def set_model_outPath(self):
        uiUtils.openFileDialog(self, self.outModel_line, "Load Trained Model", "pt")

    def add_tree_item(self, treeWidget, label, jnt_mode):
        uiUtils.add_selection(treeWidget, jnt_mode)
        uiUtils.update_param_label(treeWidget, label)

    def show_anim_context_menu(self, pos):
        uiUtils.show_context_menu(self, pos, self.anim_tree)
        self.anim_tree.checkIfEmpty()
        uiUtils.update_param_label(self.anim_tree, self.anim_param_label)


    def map_prediction(self):
        import apply_prediction
        reload(apply_prediction)
        apply_prediction.map_data()



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


    def __init__(self, parent=uiUtils.maya_main_window()):
        super(IRM_UI, self).__init__(parent)

        self.setWindowTitle("Inverse Rig Mapping Tool")
        self.resize(1000, 700)

        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint)

        self.new_window = None

        self.create_widgets()
        self.create_layouts()
        self.create_connections()


    def create_widgets(self):
        # menu bar
        self.menuBar = QtWidgets.QMenuBar()
        self.file_menu = QtWidgets.QMenu("File", self)
        self.menu_load = QtWidgets.QAction("Load Config", self)
        self.menu_save = QtWidgets.QAction("Save Config", self)
        self.menu_recent = QtWidgets.QMenu("Recent Configs", self)

        self.file_menu.addAction(self.menu_load)
        self.file_menu.addAction(self.menu_save)
        self.file_menu.addSeparator()
        #self.file_menu.addMenu(self.menu_recent)
        self.menuBar.addMenu(self.file_menu)

        # tab widgets
        self.dataGen_wdg = DataGenWidget()
        self.predict_wdg = PredictWidget()

        self.tab_widget = QtWidgets.QTabWidget(self)
        self.tab_widget.addTab(self.dataGen_wdg, "Training Setup")
        self.tab_widget.addTab(self.predict_wdg, "Predict Animation")


    def create_layouts(self):      
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(3,3,3,3)
        main_layout.addWidget(self.menuBar)
        main_layout.addWidget(self.tab_widget)


    def create_connections(self):
        self.menu_load.triggered.connect(self.load_config)
        self.menu_save.triggered.connect(self.save_config)


    def load_config(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open IRM Config", QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation), "JSON Files (*.json)")
        if file_name:
            pass


    def save_config(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save IRM Config", QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation), "JSON Files (*.json)")
        if file_name:
            pass

    def add_to_recent_configs(self, file_name):
        # Load the current list of recent configs
        try:
            with open('recent_configs.json', 'r') as f:
                recent_configs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            recent_configs = []

        # Add the new config to the front of the list, removing any duplicates
        recent_configs = [file_name] + [config for config in recent_configs if config != file_name]
        
        # Keep the list at a maximum of 5 items
        recent_configs = recent_configs[:5]

        # Write the list back to the file
        with open('recent_configs.json', 'w') as f:
            json.dump(recent_configs, f)

        self.update_recent_configs()


    def update_recent_configs(self):
        # Clear the current menu
        self.menu_recent.clear()

        # Load the list of recent configs
        try:
            with open('recent_configs.json', 'r') as f:
                recent_configs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        # Add each config to the menu
        for config in recent_configs:
            action = self.menu_recent.addAction(config)
            action.triggered.connect(lambda config=config: self.load_recent_config(config))


    def load_recent_config(self, file_name=None):
        if file_name is None:  # If no file name was provided, open a file dialog
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON", QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation), "JSON Files (*.json)")
        if file_name:
            # Load the config
            self.add_to_recent_configs(file_name)



try:
    irm_dialog.close()
    irm_dialog.deleteLater()
except:
    pass

irm_dialog = IRM_UI()
irm_dialog.show()
