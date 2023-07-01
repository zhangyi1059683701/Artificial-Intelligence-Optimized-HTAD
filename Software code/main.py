from PySide2.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QVBoxLayout, QTableView, QLabel,QScrollArea
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Qt, QFile, QModelIndex, QAbstractTableModel
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QVBoxLayout, QTableView
from mainwindow import Ui_Form
from bayes_opt import BayesianOptimization
from PIL import Image
from PIL import ImageQt
import ast
import sys
import h2o
h2o.init()
import os
import pandas as pd
import numpy as np

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(825, 595)
        Form.setStyleSheet(u"")
        self.listView = QListView(Form)
        self.listView.setObjectName(u"listView")
        self.listView.setGeometry(QRect(0, -10, 981, 621))
        self.listView.setStyleSheet(u"background-image: url(\"111.jpg\");\n"
"border: none;")
        self.tableView = QTableView(Form)
        self.tableView.setObjectName(u"tableView")
        self.tableView.setGeometry(QRect(0, 30, 341, 231))
        self.tableView.setStyleSheet(u"background-color: solid gery;\n"
"border: 1px solid black; \n"
"")
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(360, 20, 101, 41))
        font = QFont()
        font.setFamily(u"Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet(u"background-color:  #00C8EF;\n"
"border: 2px solid black;")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(60, 10, 161, 16))
        self.label.setFont(font)
        self.label.setStyleSheet(u"background-color: grey;\n"
"border: 1px solid black; ")
        self.tableView_2 = QTableView(Form)
        self.tableView_2.setObjectName(u"tableView_2")
        self.tableView_2.setGeometry(QRect(470, 170, 331, 91))
        self.tableView_2.setStyleSheet(u"\n"
"border: 1px solid black; \n"
"")
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(550, 140, 171, 21))
        self.label_2.setFont(font)
        self.label_2.setStyleSheet(u"background-color: grey;\n"
"border: 1px solid black; ")
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(360, 140, 101, 41))
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet(u"background-color: #00C8EF;\n"
"border: 2px solid black;")
        self.line = QFrame(Form)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(0, 280, 811, 16))
        self.line.setStyleSheet(u"border-top: 1.5px solid black;")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(360, 290, 151, 23))
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet(u"background-color: #C25E62;\n"
"border: 2px solid black;")
        self.line_2 = QFrame(Form)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(0, 320, 811, 16))
        self.line_2.setStyleSheet(u"border-top: 1.5px solid black;")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.pushButton_4 = QPushButton(Form)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(560, 290, 221, 23))
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet(u"background-color: #C25E62;\n"
"border: 2px solid black;")
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 290, 121, 21))
        self.label_3.setFont(font)
        self.label_3.setStyleSheet(u"background-color: grey;\n"
"border: 1px solid black; ")
        self.pushButton_5 = QPushButton(Form)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(360, 80, 101, 41))
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet(u"background-color:  #00C8EF;\n"
"border: 2px solid black;")
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(560, 6, 161, 20))
        self.label_4.setFont(font)
        self.label_4.setStyleSheet(u"background-color: grey;\n"
"border: 1px solid black; ")
        self.tableView_3 = QTableView(Form)
        self.tableView_3.setObjectName(u"tableView_3")
        self.tableView_3.setGeometry(QRect(470, 30, 331, 101))
        self.tableView_3.setStyleSheet(u"\n"
"border: 1px solid black; \n"
"")
        self.line_3 = QFrame(Form)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(180, 320, 20, 161))
        self.line_3.setLineWidth(2)
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)
        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(20, 330, 161, 21))
        self.label_5.setFont(font)
        self.label_5.setStyleSheet(u"background-color:  grey;\n"
"border: 1px solid black; ")
        self.label_6 = QLabel(Form)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 370, 71, 16))
        self.label_6.setFont(font)
        self.label_7 = QLabel(Form)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(10, 410, 51, 16))
        self.label_7.setFont(font)
        self.plainTextEdit = QPlainTextEdit(Form)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(90, 370, 91, 21))
        self.plainTextEdit_2 = QPlainTextEdit(Form)
        self.plainTextEdit_2.setObjectName(u"plainTextEdit_2")
        self.plainTextEdit_2.setGeometry(QRect(90, 410, 91, 21))
        self.label_8 = QLabel(Form)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(200, 330, 171, 21))
        self.label_8.setFont(font)
        self.label_8.setStyleSheet(u"background-color:  grey;\n"
"border: 1px solid black; ")
        self.label_9 = QLabel(Form)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(200, 360, 171, 16))
        self.label_9.setFont(font)
        self.plainTextEdit_3 = QPlainTextEdit(Form)
        self.plainTextEdit_3.setObjectName(u"plainTextEdit_3")
        self.plainTextEdit_3.setGeometry(QRect(370, 360, 101, 21))
        self.label_10 = QLabel(Form)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(200, 390, 121, 16))
        self.label_10.setFont(font)
        self.plainTextEdit_4 = QPlainTextEdit(Form)
        self.plainTextEdit_4.setObjectName(u"plainTextEdit_4")
        self.plainTextEdit_4.setGeometry(QRect(370, 390, 101, 21))
        self.label_11 = QLabel(Form)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(200, 420, 71, 16))
        self.label_11.setFont(font)
        self.plainTextEdit_5 = QPlainTextEdit(Form)
        self.plainTextEdit_5.setObjectName(u"plainTextEdit_5")
        self.plainTextEdit_5.setGeometry(QRect(370, 420, 101, 21))
        self.label_12 = QLabel(Form)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(200, 450, 111, 16))
        self.label_12.setFont(font)
        self.plainTextEdit_6 = QPlainTextEdit(Form)
        self.plainTextEdit_6.setObjectName(u"plainTextEdit_6")
        self.plainTextEdit_6.setGeometry(QRect(370, 450, 101, 21))
        self.label_13 = QLabel(Form)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(510, 330, 131, 16))
        self.label_13.setFont(font)
        self.plainTextEdit_7 = QPlainTextEdit(Form)
        self.plainTextEdit_7.setObjectName(u"plainTextEdit_7")
        self.plainTextEdit_7.setGeometry(QRect(690, 330, 101, 21))
        self.label_14 = QLabel(Form)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(510, 360, 131, 16))
        self.label_14.setFont(font)
        self.plainTextEdit_8 = QPlainTextEdit(Form)
        self.plainTextEdit_8.setObjectName(u"plainTextEdit_8")
        self.plainTextEdit_8.setGeometry(QRect(690, 360, 101, 21))
        self.label_15 = QLabel(Form)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(510, 390, 131, 16))
        self.label_15.setFont(font)
        self.plainTextEdit_9 = QPlainTextEdit(Form)
        self.plainTextEdit_9.setObjectName(u"plainTextEdit_9")
        self.plainTextEdit_9.setGeometry(QRect(690, 390, 101, 21))
        self.label_16 = QLabel(Form)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(510, 420, 131, 16))
        self.label_16.setFont(font)
        self.plainTextEdit_10 = QPlainTextEdit(Form)
        self.plainTextEdit_10.setObjectName(u"plainTextEdit_10")
        self.plainTextEdit_10.setGeometry(QRect(690, 420, 101, 21))
        self.label_17 = QLabel(Form)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(510, 450, 131, 16))
        self.label_17.setFont(font)
        self.plainTextEdit_11 = QPlainTextEdit(Form)
        self.plainTextEdit_11.setObjectName(u"plainTextEdit_11")
        self.plainTextEdit_11.setGeometry(QRect(690, 450, 101, 21))
        self.line_4 = QFrame(Form)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setGeometry(QRect(0, 480, 811, 16))
        self.line_4.setStyleSheet(u"border-top: 1.5px solid black;")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)
        self.label_18 = QLabel(Form)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(300, 490, 171, 21))
        self.label_18.setFont(font)
        self.label_18.setStyleSheet(u"background-color:  grey;\n"
"border: 1px solid black; ")
        self.textEdit = QTextEdit(Form)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(10, 520, 801, 71))
        self.textEdit.setStyleSheet(u"border: 1px solid black; ")
        self.pushButton_6 = QPushButton(Form)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(360, 200, 101, 41))
        self.pushButton_6.setFont(font)
        self.pushButton_6.setStyleSheet(u"background-color: #00C8EF;\n"
"border: 2px solid black;")
        self.pushButton_7 = QPushButton(Form)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(160, 290, 151, 23))
        self.pushButton_7.setFont(font)
        self.pushButton_7.setStyleSheet(u"background-color: #C25E62;\n"
"border: 2px solid black;")

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"水热预处理生物质厌氧消化沼气性能的预测和工艺优化智能系统V1.0", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"Import Data", None))
        self.label.setText(QCoreApplication.translate("Form", u"Input Data Visualization", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Prediction Results Show", None))
        self.pushButton_2.setText(QCoreApplication.translate("Form", u"Model Predict", None))
        self.pushButton_3.setText(QCoreApplication.translate("Form", u"Model Fitting Diagram", None))
        self.pushButton_4.setText(QCoreApplication.translate("Form", u"Features Important Diagram", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Best Model Results", None))
        self.pushButton_5.setText(QCoreApplication.translate("Form", u"Data Filling", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"Filling Results Show", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"Bayesian model parameters", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"init_points", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"n_iter", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"Define optimization interval", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"Hydrothermal temperature", None))
        self.plainTextEdit_3.setPlaceholderText(QCoreApplication.translate("Form", u"\u2103", None))
        self.label_10.setText(QCoreApplication.translate("Form", u"Hydrothermal time", None))
        self.plainTextEdit_4.setPlaceholderText(QCoreApplication.translate("Form", u"min", None))
        self.label_11.setText(QCoreApplication.translate("Form", u"Particle size", None))
        self.plainTextEdit_5.setPlaceholderText(QCoreApplication.translate("Form", u"mm", None))
        self.label_12.setText(QCoreApplication.translate("Form", u"Solid to liquid ratio", None))
        self.plainTextEdit_6.setPlaceholderText(QCoreApplication.translate("Form", u"1", None))
        self.label_13.setText(QCoreApplication.translate("Form", u"Anaerobic temperature", None))
        self.plainTextEdit_7.setPlaceholderText(QCoreApplication.translate("Form", u"\u2103", None))
        self.label_14.setText(QCoreApplication.translate("Form", u"Anaerobic time", None))
        self.plainTextEdit_8.setPlaceholderText(QCoreApplication.translate("Form", u"d", None))
        self.label_15.setText(QCoreApplication.translate("Form", u"Lignin", None))
        self.plainTextEdit_9.setPlaceholderText(QCoreApplication.translate("Form", u"%", None))
        self.label_16.setText(QCoreApplication.translate("Form", u"Cellulose", None))
        self.plainTextEdit_10.setPlaceholderText(QCoreApplication.translate("Form", u"%", None))
        self.label_17.setText(QCoreApplication.translate("Form", u"Hemicellulose", None))
        self.plainTextEdit_11.setPlaceholderText(QCoreApplication.translate("Form", u"%", None))
        self.label_18.setText(QCoreApplication.translate("Form", u"Bayesian optimization results", None))
        self.pushButton_6.setText(QCoreApplication.translate("Form", u"Optimization", None))
        self.pushButton_7.setText(QCoreApplication.translate("Form", u"Data Display  Diagram", None))
class mywindow(QWidget):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 创建一个按钮来加载CSV文件
        self.load_csv_button = self.ui.pushButton
        self.load_csv_button.clicked.connect(self.load_csv)
        self.ui.pushButton_5.clicked.connect(self.data_filling)
        self.ui.pushButton_2.clicked.connect(self.model_predict)
        # 创建一个QTableView来展示CSV数据
        self.table_view = self.ui.tableView
        self.table_view_1 = self.ui.tableView_3
        self.table_view_2 = self.ui.tableView_2
        # 将UI界面添加到窗口
        self._data = None
        self._model = None
        self.ui.pushButton_7.clicked.connect(self.show_data_image)
        self.ui.pushButton_3.clicked.connect(self.show_predict_image)
        self.ui.pushButton_4.clicked.connect(self.show_shap_image)
        self.ui.pushButton_6.clicked.connect(self.bayes_optimiza)

    def load_csv(self):
        # 使用QFileDialog来选择CSV文件
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("CSV files (*.csv)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            # 从文件中读取数据并显示在QTableView中
            filename = file_dialog.selectedFiles()[0]
            self._data = pd.read_csv(filename)
            model = PandasModel(self._data)
            self.table_view.setModel(model)

    def data_filling(self):
        from sklearn.experimental import enable_iterative_imputer  # 在现有版本下0.22，必须激活这个函数才可以使用IterativeImputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor
        import pandas as pd
        software_test_nan = self._data
        data = pd.read_csv("./imputer_rf_data.csv")
        data = data.iloc[:, 1:]
        RF = RandomForestRegressor()
        imputer_rf = IterativeImputer(max_iter=30,
                                      random_state=1,
                                      estimator=RF)

        imputer_rf.fit(data)  # 在训练集上训练

        # 转换填补训练集与测试集
        imputer_rf_data = imputer_rf.transform(software_test_nan)  # 填补数据
        imputer_rf_data = pd.DataFrame(imputer_rf_data, columns=data.columns)
        self._data = imputer_rf_data
        model = PandasModel(imputer_rf_data)
        self.table_view_1.setModel(model)

    def model_predict(self):

        path = os.path.abspath("./Best model/GBM_grid_1_AutoML_3_20230310_90755_model_125")
        best_model = h2o.load_model(path)
        software_test_nan = self._data
        software_test_nan = np.array(software_test_nan)[:, :-1]
        software_test_nan = pd.DataFrame(software_test_nan, columns=self._data.columns[:-1])
        software_test_nan = h2o.H2OFrame.from_python(software_test_nan)
        preds_software_test_nan = best_model.predict(software_test_nan)
        preds_software_test = h2o.as_list(preds_software_test_nan)
        preds_software_test.columns = ["Predict Cumulative methane production"]
        model = PandasModel(preds_software_test)
        self.table_view_2.setModel(model)
        self._model=best_model

    def show_data_image(self):
        filename = "./Figure/1.png"
        image = Image.open(filename)
        new_window = QMainWindow(self)
        new_window.setWindowTitle("Image")
        scroll = QScrollArea(new_window)
        label = QLabel(scroll)
        pixmap = QPixmap.fromImage(ImageQt.ImageQt(image))
        label.setPixmap(pixmap)
        scroll.setWidget(label)
        new_window.setCentralWidget(scroll)
        new_window.show()

    def show_predict_image(self):
        filename = "./Figure/2.png"
        image = Image.open(filename)
        new_window = QMainWindow(self)
        new_window.setWindowTitle("Image")
        scroll = QScrollArea(new_window)
        label = QLabel(scroll)
        pixmap = QPixmap.fromImage(ImageQt.ImageQt(image))
        label.setPixmap(pixmap)
        scroll.setWidget(label)
        new_window.setCentralWidget(scroll)
        new_window.show()

    def show_shap_image(self):
        filename = "./Figure/3.png"
        image = Image.open(filename)
        new_window = QMainWindow(self)
        new_window.setWindowTitle("Image")
        scroll = QScrollArea(new_window)
        label = QLabel(scroll)
        pixmap = QPixmap.fromImage(ImageQt.ImageQt(image))
        label.setPixmap(pixmap)
        scroll.setWidget(label)
        new_window.setCentralWidget(scroll)
        new_window.show()


    def bayes_optimiza(self):
        best_model = self._model

        def black_box_function(Hydrothermal_temperature, Hydrothermal_time, Particle_size, Solid_to_liquid_ratio,
                               Anaerobic_temperature, Anaerobic_time, Lignin, Cellulose, Hemicellulose):
            X = pd.DataFrame(
                np.array([Hydrothermal_temperature, Hydrothermal_time, Particle_size, Solid_to_liquid_ratio,
                          Anaerobic_temperature, Anaerobic_time, Lignin, Cellulose, Hemicellulose]).reshape(1, -1),
                columns=self._data.columns[:-1])
            X = h2o.H2OFrame.from_python(X)
            preds = best_model.predict(X)
            preds = h2o.as_list(preds)
            preds = np.squeeze(np.array(preds))
            return preds

        HTe_P = self.ui.plainTextEdit_3.toPlainText()
        HTi_P = self.ui.plainTextEdit_4.toPlainText()
        PS_P = self.ui.plainTextEdit_5.toPlainText()
        ST_P = self.ui.plainTextEdit_6.toPlainText()
        ATE_P = self.ui.plainTextEdit_7.toPlainText()
        ATi_P = self.ui.plainTextEdit_8.toPlainText()
        LI_P = self.ui.plainTextEdit_9.toPlainText()
        CE_P = self.ui.plainTextEdit_10.toPlainText()
        HE_P = self.ui.plainTextEdit_11.toPlainText()
        init_points = int(self.ui.plainTextEdit.toPlainText())
        n_iter = int(self.ui.plainTextEdit_2.toPlainText())

        pbounds = {
            'Hydrothermal_temperature': ast.literal_eval(HTe_P),
            'Hydrothermal_time': ast.literal_eval(HTi_P),
            "Particle_size": ast.literal_eval(PS_P),
            'Solid_to_liquid_ratio': ast.literal_eval(ST_P),
            'Anaerobic_temperature': ast.literal_eval(ATE_P),
            'Anaerobic_time': ast.literal_eval(ATi_P),
            'Lignin': ast.literal_eval(LI_P),
            'Cellulose': ast.literal_eval(CE_P),
            'Hemicellulose': ast.literal_eval(HE_P)}
        optimizer_rf = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent

        )
        optimizer_rf.maximize(
            init_points=init_points,  # 执行随机搜索的步数
            n_iter=n_iter  # 执行贝叶斯优化的步数
        )

        print(optimizer_rf.max)
        result = str(optimizer_rf.max)
        self.ui.textEdit.setText(result)


class PandasModel(QAbstractTableModel):
    """
    PandasModel类用于将pandas DataFrame转换为QAbstractTableModel的子类，
    以在QTableView中展示数据。
    """

    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return len(self._data.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._data.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        elif role != Qt.DisplayRole:
            return None
        else:
            row = index.row()
            col = index.column()
            return str(self._data.iloc[row, col])

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._data.columns[section]
            elif orientation == Qt.Vertical:
                return str(section + 1)
        return None



if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('logo.jpg'))
    app.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))
    # window = QMainWindow()
    # 设置窗口大小为800x600，并将其移动到屏幕左上角
    # window.setGeometry(0,0,1141,761)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())


