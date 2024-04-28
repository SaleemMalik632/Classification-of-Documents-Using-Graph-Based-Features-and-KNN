# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\Main window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1282, 663)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        MainWindow.setStyleSheet("background-color: rgb(85, 85, 127);\n"
"\n"
"\n"
"\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("QFrame {\n"
"    \n"
"    border-radius: 30px;\n"
"    background-color:black;\n"
"    border: 1px solid grey;\n"
"}\n"
"")
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMaximumSize(QtCore.QSize(1200, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        self.frame_4.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_8 = QtWidgets.QFrame(self.frame_4)
        self.frame_8.setMaximumSize(QtCore.QSize(1200, 16777215))
        self.frame_8.setStyleSheet("border:none")
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEditShowfile = QtWidgets.QLineEdit(self.frame_8)
        self.lineEditShowfile.setMinimumSize(QtCore.QSize(35, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEditShowfile.setFont(font)
        self.lineEditShowfile.setStyleSheet(" background-color: #f0f0f0; /* Background color */\n"
"    padding: 8px; /* Padding around the text */\n"
"    border-radius: 15px; /* Border radius */\n"
"    border: 2px solid #ccc; /* Border color */")
        self.lineEditShowfile.setText("")
        self.lineEditShowfile.setObjectName("lineEditShowfile")
        self.horizontalLayout_3.addWidget(self.lineEditShowfile)
        self.btnLoadFile = QtWidgets.QPushButton(self.frame_8)
        self.btnLoadFile.setMinimumSize(QtCore.QSize(200, 45))
        self.btnLoadFile.setMaximumSize(QtCore.QSize(0, 16777215))
        self.btnLoadFile.setStyleSheet("QPushButton {\n"
"    padding: 8px 16px; /* Padding around the text */\n"
"    border-radius: 20px; /* Border radius */\n"
"    border: 2px solid #ccc; /* Border color */\n"
"    background-color: rgb(16, 50, 143);\n"
"    color: rgb(255, 255, 255); /* Text color */\n"
"    background-image: url(:/icons/file_select_icons.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(0, 0, 127); /* Brighter background color on hover */\n"
"    color: rgb(255, 255, 255); /* Text color on hover */\n"
"}\n"
"")
        self.btnLoadFile.setObjectName("btnLoadFile")
        self.horizontalLayout_3.addWidget(self.btnLoadFile)
        self.horizontalLayout_2.addWidget(self.frame_8)
        self.verticalLayout.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_9 = QtWidgets.QFrame(self.frame_5)
        self.frame_9.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_9.setMaximumSize(QtCore.QSize(16777215, 72))
        self.frame_9.setStyleSheet("border:none")
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_9)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label = QtWidgets.QLabel(self.frame_9)
        self.label.setMinimumSize(QtCore.QSize(0, 9))
        self.label.setMaximumSize(QtCore.QSize(16777215, 104))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.horizontalLayout_5.addWidget(self.label)
        self.pushButton = QtWidgets.QPushButton(self.frame_9)
        self.pushButton.setMinimumSize(QtCore.QSize(200, 45))
        self.pushButton.setMaximumSize(QtCore.QSize(169, 16777215))
        self.pushButton.setStyleSheet("QPushButton {\n"
"    padding: 8px 16px; /* Padding around the text */\n"
"    border-radius: 20px; /* Border radius */\n"
"    border: 2px solid #ccc; /* Border color */\n"
"    background-color: rgb(16, 50, 143);\n"
"    color: rgb(255, 255, 255); /* Text color */\n"
"    background-image: url(:/icons/file_select_icons.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(0, 0, 127); /* Brighter background color on hover */\n"
"    color: rgb(255, 255, 255); /* Text color on hover */\n"
"}\n"
"")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.verticalLayout_3.addWidget(self.frame_9)
        self.webfram = QtWidgets.QFrame(self.frame_5)
        self.webfram.setStyleSheet("border:none")
        self.webfram.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.webfram.setFrameShadow(QtWidgets.QFrame.Raised)
        self.webfram.setObjectName("webfram")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.webfram)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_3.addWidget(self.webfram)
        self.verticalLayout.addWidget(self.frame_5)
        self.horizontalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setMinimumSize(QtCore.QSize(457, 0))
        self.frame_3.setMaximumSize(QtCore.QSize(700, 16777215))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        self.frame_6.setStyleSheet("border:none")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame_14 = QtWidgets.QFrame(self.frame_6)
        self.frame_14.setMinimumSize(QtCore.QSize(0, 292))
        self.frame_14.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_14)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_2 = QtWidgets.QLabel(self.frame_14)
        self.label_2.setMinimumSize(QtCore.QSize(0, 35))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 62))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(255, 255, 255);\n"
"border:none")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_7.addWidget(self.label_2)
        self.graphicsView_for_matrix = QtWidgets.QGraphicsView(self.frame_14)
        self.graphicsView_for_matrix.setStyleSheet("border-image: url(:/newPrefix/icons/name_showmatrix.jpeg);")
        self.graphicsView_for_matrix.setObjectName("graphicsView_for_matrix")
        self.verticalLayout_7.addWidget(self.graphicsView_for_matrix)
        self.verticalLayout_5.addWidget(self.frame_14)
        self.frame_7 = QtWidgets.QFrame(self.frame_6)
        self.frame_7.setStyleSheet("   border-radius: 30px;\n"
"    border: 1px solid grey;")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_11 = QtWidgets.QFrame(self.frame_7)
        self.frame_11.setMinimumSize(QtCore.QSize(175, 0))
        self.frame_11.setMaximumSize(QtCore.QSize(200, 16777215))
        self.frame_11.setStyleSheet("border:none")
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_11)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.PrecisionLabel = QtWidgets.QLabel(self.frame_11)
        self.PrecisionLabel.setStyleSheet("border:none;\n"
"color: rgb(255, 255, 255);")
        self.PrecisionLabel.setObjectName("PrecisionLabel")
        self.verticalLayout_6.addWidget(self.PrecisionLabel)
        self.AccuracyLabel = QtWidgets.QLabel(self.frame_11)
        self.AccuracyLabel.setStyleSheet("border:none;\n"
"color: rgb(255, 255, 255);")
        self.AccuracyLabel.setObjectName("AccuracyLabel")
        self.verticalLayout_6.addWidget(self.AccuracyLabel)
        self.F1ScoreLabel = QtWidgets.QLabel(self.frame_11)
        self.F1ScoreLabel.setStyleSheet("border:none;\n"
"color: rgb(255, 255, 255);")
        self.F1ScoreLabel.setObjectName("F1ScoreLabel")
        self.verticalLayout_6.addWidget(self.F1ScoreLabel)
        self.RecallLabel = QtWidgets.QLabel(self.frame_11)
        self.RecallLabel.setStyleSheet("border:none;\n"
"color: rgb(255, 255, 255);")
        self.RecallLabel.setObjectName("RecallLabel")
        self.verticalLayout_6.addWidget(self.RecallLabel)
        self.horizontalLayout_4.addWidget(self.frame_11)
        self.frame_12 = QtWidgets.QFrame(self.frame_7)
        self.frame_12.setMinimumSize(QtCore.QSize(10, 0))
        self.frame_12.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.frame_12.setStyleSheet("border:none")
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_12)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_8 = QtWidgets.QLabel(self.frame_12)
        self.label_8.setMinimumSize(QtCore.QSize(20, 0))
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 49))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("color: rgb(255, 255, 255);\n"
"border:none")
        self.label_8.setObjectName("label_8")
        self.verticalLayout_4.addWidget(self.label_8)
        self.graphicsView_for_accuracy = QtWidgets.QGraphicsView(self.frame_12)
        self.graphicsView_for_accuracy.setStyleSheet("border-image: url(:/newPrefix/icons/name_showmatrix.jpeg);")
        self.graphicsView_for_accuracy.setObjectName("graphicsView_for_accuracy")
        self.verticalLayout_4.addWidget(self.graphicsView_for_accuracy)
        self.horizontalLayout_4.addWidget(self.frame_12)
        self.verticalLayout_5.addWidget(self.frame_7)
        self.verticalLayout_2.addWidget(self.frame_6)
        self.horizontalLayout.addWidget(self.frame_3)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnLoadFile.setText(_translate("MainWindow", "Browse"))
        self.label.setText(_translate("MainWindow", "Docoment Graphs"))
        self.pushButton.setText(_translate("MainWindow", "Start Processing"))
        self.label_2.setText(_translate("MainWindow", "Confusion Matrix"))
        self.PrecisionLabel.setText(_translate("MainWindow", "Precision: 11.11111111111111 %"))
        self.AccuracyLabel.setText(_translate("MainWindow", "Accuracy: 33.33333333333333 %"))
        self.F1ScoreLabel.setText(_translate("MainWindow", "F1-score: 16.666666666666664 %"))
        self.RecallLabel.setText(_translate("MainWindow", "Recall: 11.11111111111111 %"))
        self.label_8.setText(_translate("MainWindow", "Accuracy Graph"))
import recources_rc
