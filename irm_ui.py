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

        self.rig_tree.checkIfEmpty()
        self.jnt_tree.checkIfEmpty()

    def create_widgets(self):
        # rig widgets   
        self.rig_param_label = QtWidgets.QLabel("Control Rig Parameters (0)")
        self.rig_param_label.setStyleSheet("color: #00ff6e;")
        self.rig_clear_btn = QtWidgets.QPushButton("Clear All")
        self.rig_add_btn = QtWidgets.QPushButton("Add")
            
        # rig tree view widget
        #self.rig_tree = QtWidgets.QTreeWidget(self)
        rig_msg = "To start generation add parameters to sample using the green + button.\nMake sure you have at least one object selected in the outliner."
        self.rig_tree = uiUtils.PlaceholderTreeWidget(self, rig_msg)
        self.rig_tree.setHeaderLabels(['Control Name', 'Min', 'Max'])
        self.rig_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.rig_tree.setItemDelegate(uiUtils.EditableItemDelegate(self.rig_tree))
        self.rig_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        header = self.rig_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header.resizeSection(1, 100)
        header.resizeSection(2, 100)


        # joint widgets
        self.jnt_param_label = QtWidgets.QLabel("Joint Parameters (0)")
        self.jnt_param_label.setStyleSheet("color: #00ff6e;")
        self.jnt_clear_btn = QtWidgets.QPushButton("Clear All")
        self.jnt_add_btn = QtWidgets.QPushButton("Add")

        # joint tree view widget
        jnt_msg = "To start generation add parameters to sample using the green + button.\nMake sure you have at least one object selected in the outliner."
        self.jnt_tree = uiUtils.PlaceholderTreeWidget(self, jnt_msg)
        self.jnt_tree.setHeaderLabels(['Joint Name'])
        self.jnt_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.jnt_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)


        # train data settings
        self.setting_widget = QtWidgets.QWidget()
        self.paramProperties_label = QtWidgets.QLabel("Parameter Properties:")

        self.minRange_label = QtWidgets.QLabel("Minimum:")
        self.minRange_line = QtWidgets.QLineEdit("-50.000")
        self.minRange_line.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("^-?\d+(\.\d{0,3})?$")))

        self.maxRange_label = QtWidgets.QLabel("Maximum:")
        self.maxRange_line = QtWidgets.QLineEdit("50.000")
        self.maxRange_line.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("^-?\d+(\.\d{0,3})?$")))

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
        self.outRig_line = QtWidgets.QLineEdit("irm_rig_data.csv")

        self.outJnt_label = QtWidgets.QLabel("Joint File:")
        self.outJnt_line = QtWidgets.QLineEdit("irm_rig_data.csv")

        self.outSpacer01_label = QtWidgets.QLabel()
        self.outSpacer01_label.setFixedHeight(10)

        self.outModel_label = QtWidgets.QLabel("Model File:")
        self.outModel_line = QtWidgets.QLineEdit("trained_model.pt")

        self.outSpacer02_label = QtWidgets.QLabel()
        self.outSpacer02_label.setFixedHeight(50)


        # buttons
        self.generate_btn = QtWidgets.QPushButton("Generate Data")
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

        output_layout.addWidget(self.outJnt_label, 2, 0)
        output_layout.addWidget(self.outJnt_line, 2, 1)

        output_layout.addWidget(self.outSpacer01_label, 3, 0)

        output_layout.addWidget(self.outModel_label, 4, 0)
        output_layout.addWidget(self.outModel_line, 4, 1)

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
        self.rig_clear_btn.clicked.connect(partial(self.clear_tree, self.rig_tree, self.rig_param_label))
        self.rig_add_btn.clicked.connect(partial(self.add_tree_item, self.rig_tree, self.rig_param_label))
        
        self.rig_tree.customContextMenuRequested.connect(self.show_rig_context_menu)
        self.rig_tree.itemChanged.connect(self.rig_tree.checkIfEmpty)
        self.rig_tree.itemSelectionChanged.connect(self.update_param_range)

        # joint UI
        self.jnt_clear_btn.clicked.connect(partial(self.clear_tree, self.jnt_tree, self.jnt_param_label))
        self.jnt_add_btn.clicked.connect(partial(self.add_tree_item, self.jnt_tree, self.jnt_param_label))
        
        self.jnt_tree.customContextMenuRequested.connect(self.show_jnt_context_menu)
        self.jnt_tree.itemChanged.connect(self.jnt_tree.checkIfEmpty)

        # data/train settings
        self.minRange_line.editingFinished.connect(self.update_min_range)

        self.maxRange_line.editingFinished.connect(self.update_max_range)

        #self.numPoses_line.editingFinished.connect(partial(self.ensure_notZero, self.numPoses_line))

        self.generate_btn.clicked.connect(self.generate_train_data)
        self.train_btn.clicked.connect(self.train_model)
    

    def add_tree_item(self, treeWidget, label):
        uiUtils.add_selection(treeWidget)
        self.update_param_label(treeWidget, label)

    def update_param_label(self, treeWidget, label):
        count = 0
        for i in range(0, treeWidget.topLevelItemCount()):
            parent = treeWidget.topLevelItem(i)
            count += parent.childCount()

        label.setText("{}({})".format(label.text().rpartition("(")[0], count))


    def clear_tree(self, treeWidget, label):
        treeWidget.clear()
        treeWidget.checkIfEmpty()
        self.update_param_label(treeWidget, label)


    def show_rig_context_menu(self, pos):
        self.show_context_menu(pos, self.rig_tree)
        self.rig_tree.checkIfEmpty()
        self.update_param_label(self.rig_tree, self.rig_param_label)

    def show_jnt_context_menu(self, pos):
        self.show_context_menu(pos, self.jnt_tree)
        self.jnt_tree.checkIfEmpty()
        self.update_param_label(self.jnt_tree, self.jnt_param_label)

        
    def show_context_menu(self, pos, treeWidget):
        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(partial(self.delete_items, treeWidget))
        menu.exec_(treeWidget.viewport().mapToGlobal(pos))


    def delete_items(self, treeWidget):
        selected_items = treeWidget.selectedItems()
        for item in selected_items:
            (item.parent() or treeWidget.invisibleRootItem()).removeChild(item)


    def update_param_range(self):
        selected_items = self.rig_tree.selectedItems()

        if selected_items:
            item = selected_items[0]

            if item.childCount() > 0:
                return
            else:
                self.minRange_line.setText(item.text(1))
                self.maxRange_line.setText(item.text(2))

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

        self.setWindowTitle("Inverse Rig Mapping Tool")
        self.resize(1000, 700)

        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint)

        self.new_window = None

        self.create_widgets()
        self.create_layouts()
        self.create_connections()

    def create_widgets(self):
        self.dataGen_wdg = DataGenWidget()
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
