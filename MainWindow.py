# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/pic/src/icons/logo.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("")
        MainWindow.setIconSize(QtCore.QSize(24, 24))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_top = QtWidgets.QFrame(self.centralwidget)
        self.frame_top.setMinimumSize(QtCore.QSize(600, 45))
        self.frame_top.setMaximumSize(QtCore.QSize(16777215, 45))
        self.frame_top.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.frame_top.setStyleSheet("background-color: rgb(60, 60, 60);")
        self.frame_top.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_top.setLineWidth(0)
        self.frame_top.setObjectName("frame_top")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_top)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_top_left = QtWidgets.QFrame(self.frame_top)
        self.frame_top_left.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top_left.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top_left.setObjectName("frame_top_left")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_top_left)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.button_open_image = QtWidgets.QPushButton(self.frame_top_left)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_open_image.sizePolicy().hasHeightForWidth())
        self.button_open_image.setSizePolicy(sizePolicy)
        self.button_open_image.setMinimumSize(QtCore.QSize(45, 45))
        self.button_open_image.setMaximumSize(QtCore.QSize(45, 45))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.button_open_image.setFont(font)
        self.button_open_image.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.button_open_image.setToolTipDuration(-1)
        self.button_open_image.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.button_open_image.setStyleSheet("QPushButton {\n"
"    border: 0px solid;\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(140, 140, 140);\n"
"}")
        self.button_open_image.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/pic/src/icons/open.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_open_image.setIcon(icon1)
        self.button_open_image.setIconSize(QtCore.QSize(24, 24))
        self.button_open_image.setObjectName("button_open_image")
        self.horizontalLayout_4.addWidget(self.button_open_image, 0, QtCore.Qt.AlignLeft)
        self.button_save_image = QtWidgets.QPushButton(self.frame_top_left)
        self.button_save_image.setEnabled(True)
        self.button_save_image.setMinimumSize(QtCore.QSize(45, 45))
        self.button_save_image.setMaximumSize(QtCore.QSize(45, 45))
        self.button_save_image.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.button_save_image.setToolTipDuration(-1)
        self.button_save_image.setStyleSheet("QPushButton {\n"
"    border: 0px solid;\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(140, 140, 140);\n"
"}\n"
"QPushButton:disabled {\n"
"    qproperty-icon: url(:/pic/src/icons/save_disabled.svg)\n"
"}")
        self.button_save_image.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/pic/src/icons/save.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap(":/pic/src/icons/save_disabled.svg"), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        self.button_save_image.setIcon(icon2)
        self.button_save_image.setIconSize(QtCore.QSize(24, 24))
        self.button_save_image.setObjectName("button_save_image")
        self.horizontalLayout_4.addWidget(self.button_save_image, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout_3.addWidget(self.frame_top_left, 0, QtCore.Qt.AlignLeft)
        self.button_run = QtWidgets.QPushButton(self.frame_top)
        self.button_run.setEnabled(True)
        self.button_run.setMinimumSize(QtCore.QSize(90, 45))
        self.button_run.setMaximumSize(QtCore.QSize(90, 45))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.button_run.setFont(font)
        self.button_run.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.button_run.setToolTipDuration(-1)
        self.button_run.setStyleSheet("QPushButton {\n"
"    border: 0px solid;\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(140, 140, 140);\n"
"}\n"
"QPushButton:disabled {\n"
"    color: rgb(100, 100, 100);\n"
"}")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/pic/src/icons/run.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3.addPixmap(QtGui.QPixmap(":/pic/src/icons/run_disabled.svg"), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        self.button_run.setIcon(icon3)
        self.button_run.setIconSize(QtCore.QSize(19, 19))
        self.button_run.setObjectName("button_run")
        self.horizontalLayout_3.addWidget(self.button_run)
        self.verticalLayout.addWidget(self.frame_top)
        self.frame_main = QtWidgets.QFrame(self.centralwidget)
        self.frame_main.setStyleSheet("QScrollBar:horizontal\n"
"{\n"
"    height: 15px;\n"
"    margin: 3px 3px 3px 3px;\n"
"    border-radius: 4px;\n"
"    background-color: none;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal\n"
"{\n"
"    background-color: rgb(135, 135, 135);\n"
"    min-width: 4px;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal:hover\n"
"{\n"
"    background-color: rgb(100, 100, 100);\n"
"    min-width: 4px;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal\n"
"{\n"
"    margin: 0px 0px 0px 0px;\n"
"    width: 0px;\n"
"    height: 0px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal\n"
"{\n"
"    margin: 0px 0px 0px 0px;\n"
"    width: 0px;\n"
"    height: 0px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal\n"
"{\n"
"    background: none;\n"
"}\n"
"\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"    background: none;\n"
"}\n"
"\n"
"QScrollBar:vertical\n"
"{\n"
"    width: 15px;\n"
"    margin: 3px 3px 3px 3px;\n"
"    border-radius: 4px;\n"
"    background-color: none;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical\n"
"{\n"
"    background-color: rgb(135, 135, 135);\n"
"    min-height: 4px;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical:hover\n"
"{\n"
"    background-color: rgb(100, 100, 100);\n"
"    min-height: 4px;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical\n"
"{\n"
"    margin: 0px 0px 0px 0px;\n"
"    width: 0px;\n"
"    height: 0px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical\n"
"{\n"
"    margin: 0px 0px 0px 0px;\n"
"    width: 0px;\n"
"    height: 0px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical\n"
"{\n"
"    background: none;\n"
"}\n"
"\n"
"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical\n"
"{\n"
"    background: none;\n"
"}")
        self.frame_main.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_main.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_main.setLineWidth(0)
        self.frame_main.setObjectName("frame_main")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_main)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_left = QtWidgets.QFrame(self.frame_main)
        self.frame_left.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.frame_left.setStyleSheet("background-color: rgb(220, 220, 220);")
        self.frame_left.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_left.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left.setLineWidth(0)
        self.frame_left.setObjectName("frame_left")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_left)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.frame_left)
        self.scrollArea.setStyleSheet("")
        self.scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea.setLineWidth(0)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 550, 555))
        self.scrollAreaWidgetContents.setStyleSheet("")
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_image = DrawLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_image.sizePolicy().hasHeightForWidth())
        self.label_image.setSizePolicy(sizePolicy)
        self.label_image.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(20)
        self.label_image.setFont(font)
        self.label_image.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_image.setAutoFillBackground(False)
        self.label_image.setStyleSheet("background-color: rgb(220, 220, 220);")
        self.label_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label_image.setLineWidth(1)
        self.label_image.setText("")
        self.label_image.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_image.setObjectName("label_image")
        self.horizontalLayout_7.addWidget(self.label_image)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout_2.addWidget(self.scrollArea)
        self.horizontalLayout.addWidget(self.frame_left)
        self.frame_right = QtWidgets.QFrame(self.frame_main)
        self.frame_right.setMinimumSize(QtCore.QSize(250, 555))
        self.frame_right.setMaximumSize(QtCore.QSize(250, 16777215))
        self.frame_right.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.frame_right.setStyleSheet("background-color: rgb(100, 100, 100);")
        self.frame_right.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_right.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_right.setLineWidth(0)
        self.frame_right.setObjectName("frame_right")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_right)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_right_top = QtWidgets.QFrame(self.frame_right)
        self.frame_right_top.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_right_top.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_right_top.setObjectName("frame_right_top")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_right_top)
        self.verticalLayout_3.setContentsMargins(10, -1, 10, -1)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_size = QtWidgets.QLabel(self.frame_right_top)
        self.label_size.setMinimumSize(QtCore.QSize(0, 30))
        self.label_size.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_size.setFont(font)
        self.label_size.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_size.setObjectName("label_size")
        self.verticalLayout_3.addWidget(self.label_size, 0, QtCore.Qt.AlignTop)
        self.frame_size = QtWidgets.QFrame(self.frame_right_top)
        self.frame_size.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_size.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_size.setObjectName("frame_size")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_size)
        self.horizontalLayout_5.setContentsMargins(8, 3, 0, 2)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_width = QtWidgets.QLabel(self.frame_size)
        self.label_width.setMinimumSize(QtCore.QSize(45, 25))
        self.label_width.setMaximumSize(QtCore.QSize(45, 25))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.label_width.setFont(font)
        self.label_width.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_width.setObjectName("label_width")
        self.horizontalLayout_5.addWidget(self.label_width)
        self.edit_width = QtWidgets.QLineEdit(self.frame_size)
        self.edit_width.setEnabled(True)
        self.edit_width.setMinimumSize(QtCore.QSize(60, 25))
        self.edit_width.setMaximumSize(QtCore.QSize(60, 25))
        self.edit_width.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.edit_width.setStyleSheet("QLineEdit {\n"
"    color: rgb(255, 255, 255);\n"
"    border: 0.5px solid rgb(200, 200, 200);\n"
"    font: 9pt \"Segoe UI\";\n"
"}\n"
"QLineEdit:focus {\n"
"    background-color: rgb(70, 70, 70);\n"
"}\n"
"QLineEdit:disabled {\n"
"    background-color: rgb(120, 120, 120);\n"
"}\n"
"QMenu {\n"
"    background-color: rgb(225, 225, 225);\n"
"}")
        self.edit_width.setMaxLength(4)
        self.edit_width.setClearButtonEnabled(False)
        self.edit_width.setObjectName("edit_width")
        self.horizontalLayout_5.addWidget(self.edit_width)
        self.label_height = QtWidgets.QLabel(self.frame_size)
        self.label_height.setMinimumSize(QtCore.QSize(45, 25))
        self.label_height.setMaximumSize(QtCore.QSize(45, 25))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.label_height.setFont(font)
        self.label_height.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_height.setObjectName("label_height")
        self.horizontalLayout_5.addWidget(self.label_height, 0, QtCore.Qt.AlignRight)
        self.edit_height = QtWidgets.QLineEdit(self.frame_size)
        self.edit_height.setMinimumSize(QtCore.QSize(60, 25))
        self.edit_height.setMaximumSize(QtCore.QSize(60, 25))
        self.edit_height.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.edit_height.setStyleSheet("QLineEdit {\n"
"    color: rgb(255, 255, 255);\n"
"    border: 0.5px solid rgb(200, 200, 200);\n"
"    font: 9pt \"Segoe UI\";\n"
"}\n"
"QLineEdit:focus {\n"
"    background-color: rgb(70, 70, 70);\n"
"}\n"
"QLineEdit:disabled {\n"
"    background-color: rgb(120, 120, 120);\n"
"}\n"
"QMenu {\n"
"    background-color: rgb(225, 225, 225);\n"
"}")
        self.edit_height.setMaxLength(4)
        self.edit_height.setObjectName("edit_height")
        self.horizontalLayout_5.addWidget(self.edit_height, 0, QtCore.Qt.AlignRight)
        self.label_width.raise_()
        self.label_height.raise_()
        self.edit_height.raise_()
        self.edit_width.raise_()
        self.verticalLayout_3.addWidget(self.frame_size)
        self.label_remove_or_keep = QtWidgets.QLabel(self.frame_right_top)
        self.label_remove_or_keep.setMinimumSize(QtCore.QSize(0, 30))
        self.label_remove_or_keep.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_remove_or_keep.setFont(font)
        self.label_remove_or_keep.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_remove_or_keep.setLineWidth(1)
        self.label_remove_or_keep.setIndent(0)
        self.label_remove_or_keep.setObjectName("label_remove_or_keep")
        self.verticalLayout_3.addWidget(self.label_remove_or_keep)
        self.frame_options = QtWidgets.QFrame(self.frame_right_top)
        self.frame_options.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_options.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_options.setObjectName("frame_options")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_options)
        self.verticalLayout_5.setContentsMargins(8, 2, 0, 3)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.checkBox_face = QtWidgets.QCheckBox(self.frame_options)
        self.checkBox_face.setMinimumSize(QtCore.QSize(0, 25))
        self.checkBox_face.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.checkBox_face.setFont(font)
        self.checkBox_face.setStyleSheet("QCheckBox{\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"    color: rgb(20, 20, 20);\n"
"}")
        self.checkBox_face.setIconSize(QtCore.QSize(16, 16))
        self.checkBox_face.setChecked(False)
        self.checkBox_face.setAutoExclusive(False)
        self.checkBox_face.setObjectName("checkBox_face")
        self.verticalLayout_5.addWidget(self.checkBox_face)
        self.checkBox_keep_objects = QtWidgets.QCheckBox(self.frame_options)
        self.checkBox_keep_objects.setMinimumSize(QtCore.QSize(0, 25))
        self.checkBox_keep_objects.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.checkBox_keep_objects.setFont(font)
        self.checkBox_keep_objects.setStyleSheet("QCheckBox{\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"    color: rgb(20, 20, 23);\n"
"}")
        self.checkBox_keep_objects.setAutoExclusive(False)
        self.checkBox_keep_objects.setObjectName("checkBox_keep_objects")
        self.verticalLayout_5.addWidget(self.checkBox_keep_objects)
        self.checkBox_remove_objects = QtWidgets.QCheckBox(self.frame_options)
        self.checkBox_remove_objects.setMinimumSize(QtCore.QSize(0, 25))
        self.checkBox_remove_objects.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.checkBox_remove_objects.setFont(font)
        self.checkBox_remove_objects.setStyleSheet("QCheckBox{\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"    color: rgb(20, 20, 20);\n"
"}")
        self.checkBox_remove_objects.setAutoExclusive(False)
        self.checkBox_remove_objects.setObjectName("checkBox_remove_objects")
        self.verticalLayout_5.addWidget(self.checkBox_remove_objects)
        self.verticalLayout_3.addWidget(self.frame_options)
        self.verticalLayout_2.addWidget(self.frame_right_top, 0, QtCore.Qt.AlignTop)
        self.horizontalLayout.addWidget(self.frame_right)
        self.verticalLayout.addWidget(self.frame_main)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Resizeing Tool"))
        self.button_open_image.setToolTip(_translate("MainWindow", "Open Image"))
        self.button_open_image.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.button_save_image.setToolTip(_translate("MainWindow", "Save Image"))
        self.button_save_image.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.button_run.setToolTip(_translate("MainWindow", "Resize Image"))
        self.button_run.setText(_translate("MainWindow", "Process"))
        self.button_run.setShortcut(_translate("MainWindow", "Ctrl+Return"))
        self.label_size.setText(_translate("MainWindow", "Size"))
        self.label_width.setText(_translate("MainWindow", "Width"))
        self.label_height.setText(_translate("MainWindow", "Height"))
        self.label_remove_or_keep.setText(_translate("MainWindow", "Options"))
        self.checkBox_face.setText(_translate("MainWindow", "Keep Face"))
        self.checkBox_keep_objects.setText(_translate("MainWindow", "Keep Objects"))
        self.checkBox_remove_objects.setText(_translate("MainWindow", "Remove Objects"))
from CustomizedWidgets import DrawLabel
import img_src_rc
