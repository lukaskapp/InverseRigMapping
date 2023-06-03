from PySide2 import QtCore, QtWidgets, QtGui
import maya.cmds as cmds
from functools import partial
from imp import reload

import utils.ui as uiUtils
reload(uiUtils)



class DataGenWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DataGenWidget, self).__init__(parent)

        self.create_widgets()
        self.create_layouts()
        self.create_connections()

    def create_widgets(self):
        # rig widgets   
        self.rig_add_btn = QtWidgets.QPushButton("Add")
            
        self.rig_tree = QtWidgets.QTreeWidget(self)
        self.rig_tree.setHeaderLabels(['Control Name', 'Min', 'Max'])
        self.rig_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.rig_tree.setItemDelegate(uiUtils.EditableItemDelegate(self.rig_tree))

        header = self.rig_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header.resizeSection(1, 100)
        header.resizeSection(2, 100)


        # joint widgets
        self.jnt_add_btn = QtWidgets.QPushButton("Add")

        self.jnt_tree = QtWidgets.QTreeWidget()
        self.jnt_tree.setHeaderLabels(['Control Name'])
        #self.jnt_table.verticalHeader().setVisible(False)
        self.jnt_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        self.setting_widget = QtWidgets.QWidget()

        self.numPoses_label = QtWidgets.QLabel("Num Poses:")
        self.numPoses_label.setFixedWidth(70)
        self.numPoses_line = QtWidgets.QLineEdit("1000")
        self.numPoses_line.setFixedWidth(60)

        self.generate_btn = QtWidgets.QPushButton("Generate")

        self.lr_label = QtWidgets.QLabel("Learning Rate:")
        self.lr_label.setFixedWidth(70)
        self.lr_line = QtWidgets.QLineEdit("0.01")
        self.lr_line.setFixedWidth(60)

        self.epoch_label = QtWidgets.QLabel("Epochs:")
        self.epoch_label.setFixedWidth(70)
        self.epoch_line = QtWidgets.QLineEdit("50")
        self.epoch_line.setFixedWidth(60)

        self.train_btn = QtWidgets.QPushButton("Train Model")


    def create_layouts(self): 
        # main UI layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(3,3,3,3)

        # left UI side - rig and joint attrs
        attr_layout = QtWidgets.QVBoxLayout(self)
        layout_size = self.size()
        layout_size.setWidth(layout_size.width() // 2)
        attr_layout.setGeometry(QtCore.QRect(self.geometry().x(), self.geometry().y(), layout_size.width(), layout_size.height()))
        main_layout.addLayout(attr_layout)

        # rig layout
        rig_group = QtWidgets.QGroupBox('Control Rig Attributes')
        attr_layout.addWidget(rig_group)
        rig_attr_layout = QtWidgets.QVBoxLayout()
        rig_group.setLayout(rig_attr_layout)

        rig_attr_layout.addWidget(self.rig_add_btn)
        rig_attr_layout.addWidget(self.rig_tree)


        # jnt layout
        jnt_group = QtWidgets.QGroupBox('Joint Attributes')
        attr_layout.addWidget(jnt_group)
        jnt_attr_layout = QtWidgets.QVBoxLayout()
        jnt_group.setLayout(jnt_attr_layout)

        jnt_attr_layout.addWidget(self.jnt_add_btn)        
        jnt_attr_layout.addWidget(self.jnt_tree)

        # right UI side - settings for data generation
        setting_layout = QtWidgets.QVBoxLayout(self)

        numPoses_layout = QtWidgets.QHBoxLayout()
        numPoses_layout.addWidget(self.numPoses_label)
        numPoses_layout.addWidget(self.numPoses_line)
        setting_layout.addLayout(numPoses_layout)

        setting_layout.addWidget(self.generate_btn)

        learningRate_layout = QtWidgets.QHBoxLayout()
        learningRate_layout.addWidget(self.lr_label)
        learningRate_layout.addWidget(self.lr_line)
        setting_layout.addLayout(learningRate_layout)

        epoch_layout = QtWidgets.QHBoxLayout()
        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addWidget(self.epoch_line)
        setting_layout.addLayout(epoch_layout)

        setting_layout.addWidget(self.train_btn)

        main_layout.addLayout(setting_layout)
        

    def create_connections(self):
        self.rig_add_btn.clicked.connect(partial(uiUtils.add_selection, self.rig_tree))
        self.rig_tree.customContextMenuRequested.connect(self.show_rig_context_menu)
        #self.rig_tree.itemDoubleClicked.connect(self.rename_item)

        self.jnt_add_btn.clicked.connect(partial(uiUtils.add_selection, self.jnt_tree))
        self.jnt_tree.customContextMenuRequested.connect(self.show_jnt_context_menu)

        self.generate_btn.clicked.connect(self.generate_train_data)
        self.train_btn.clicked.connect(self.train_model)
    

    def show_rig_context_menu(self, pos):
        self.show_context_menu(pos, self.rig_tree)

    def show_jnt_context_menu(self, pos):
        self.show_context_menu(pos, self.jnt_tree)

    def show_context_menu(self, pos, treeWidget):
        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(partial(self.delete_item, treeWidget))
        menu.exec_(treeWidget.viewport().mapToGlobal(pos))


    def delete_item(self, treeWidget):
        selected_item = treeWidget.currentItem()
        (selected_item.parent() or treeWidget.invisibleRootItem()).removeChild(selected_item)


    def rename_item(self, item):
        old_name = item.text(0)
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename", "Enter new name:", QtWidgets.QLineEdit.Normal, old_name)
        if ok and new_name != old_name:
            item.setText(0, new_name)


    def generate_train_data(self):
        rig_input_data = []
        for row in range(self.rig_table.rowCount()):
            item = self.rig_table.item(row, 0)
            if item:
                rig_input_data.append(item.text())


        jnt_input_data = []
        for row in range(self.jnt_table.rowCount()):
            item = self.jnt_table.item(row, 0)
            if item:
                jnt_input_data.append(item.text())

        print("RIG INPUT DATA: ", rig_input_data)
        print("JNT INPUT DATA: ", jnt_input_data)      


        import prep_training_data
        reload(prep_training_data)
        prep_training_data.prep_data(rig_input_data, jnt_input_data)


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

    def create_widgets(self):
        # rig widgets   
        self.rig_add_btn = QtWidgets.QPushButton("Add")

            
        self.rig_table = QtWidgets.QTableWidget(0, 1)
        self.rig_table.setHorizontalHeaderLabels(['Joint Name'])
        self.rig_table.verticalHeader().setVisible(False)

        rig_table_width = self.rig_table.geometry().width()
        #self.rig_table.setColumnWidth(0, rig_table_width * 0.98)

        #self.rig_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
        self.rig_table.horizontalHeader().setStretchLastSection(True)
        self.rig_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)



        self.setting_widget = QtWidgets.QWidget()

        self.numPoses_label = QtWidgets.QLabel("Num Poses:")
        self.numPoses_label.setFixedWidth(70)
        self.numPoses_line = QtWidgets.QLineEdit("1000")
        self.numPoses_line.setFixedWidth(60)

        self.map_btn = QtWidgets.QPushButton("Map Prediction")



    def create_layouts(self): 
        # main UI layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(3,3,3,3)


        # left UI side - rig and joint attrs
        attr_layout = QtWidgets.QVBoxLayout(self)
        layout_size = self.size()
        layout_size.setWidth(layout_size.width() // 2)
        attr_layout.setGeometry(QtCore.QRect(self.geometry().x(), self.geometry().y(), layout_size.width(), layout_size.height()))
        main_layout.addLayout(attr_layout)


        # rig layout
        rig_group = QtWidgets.QGroupBox('Animation Attributes')
        attr_layout.addWidget(rig_group)
        rig_attr_layout = QtWidgets.QVBoxLayout()
        rig_group.setLayout(rig_attr_layout)



        rig_attr_layout.addWidget(self.rig_add_btn)
        rig_attr_layout.addWidget(self.rig_table)



        # right UI side - settings for data generation
        setting_layout = QtWidgets.QVBoxLayout(self)

        numPoses_layout = QtWidgets.QHBoxLayout()
        numPoses_layout.addWidget(self.numPoses_label)
        numPoses_layout.addWidget(self.numPoses_line)
        #setting_layout.addLayout(numPoses_layout)

        setting_layout.addWidget(self.map_btn)

        main_layout.addLayout(setting_layout)

    def create_connections(self):
        self.rig_add_btn.clicked.connect(self.add_jnt_selection)
        self.map_btn.clicked.connect(self.map_predict)

    def add_jnt_selection(self):
        sel = cmds.ls(sl=1)
        for obj in sel:
            self.add_table_row(tableWidget=self.rig_table, item_list=[obj])

    def add_table_row(self, tableWidget, item_list):
        row_count = tableWidget.rowCount()
        tableWidget.setRowCount(row_count + 1)
        for i, name in enumerate(item_list):
            item = QtWidgets.QTableWidgetItem(name)
            tableWidget.setItem(row_count, i, item)

    def map_predict(self):
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

        self.setWindowTitle("IRM Tool")
        self.resize(800, 800)

        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint)

        self.new_window = None

        self.create_widgets()
        self.create_layouts()
        self.create_connections()

    def create_widgets(self):
        self.dataGen_wdg = DataGenWidget()
        self.train_wdg = TrainWidget()
        self.predict_wdg = PredictWidget()


        self.tab_widget = QtWidgets.QTabWidget(self)
        self.tab_widget.addTab(self.dataGen_wdg, "Training Setup")
        #self.tab_widget.addTab(self.train_wdg, "Train Model")
        self.tab_widget.addTab(self.predict_wdg, "Predict Animation")



    def create_layouts(self):      
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(3,3,3,3)
        main_layout.addWidget(self.tab_widget)


    def create_connections(self):
        pass



try:
    irm_dialog.close()
    irm_dialog.deleteLater()
except:
    pass

irm_dialog = IRM_UI()
irm_dialog.show()
