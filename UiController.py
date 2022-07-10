import os
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
import cv2
import numpy as np

import MainWindow
from imageData import ImageData


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Ui Settings
        self.ui = MainWindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_ui()
        # Image Data and Path
        self.__image = ImageData()
        self.__image_path, self.__image_name = None, None

    def init_ui(self):
        """ Initiate UI of Application Window. """
        # Connect Handle Functions to Widgets
        self.ui.button_run.clicked.connect(self.button_run_click)
        self.ui.button_open_image.clicked.connect(self.button_open_image_click)
        self.ui.button_save_image.clicked.connect(self.button_save_image_click)

        self.ui.checkBox_face.stateChanged.connect(self.checkbox_face_state_changed)
        self.ui.checkBox_remove_objects.stateChanged.connect(self.checkbox_remove_objects_state_changed)
        self.ui.checkBox_keep_objects.stateChanged.connect(self.checkbox_keep_objects_state_changed)
        # Set Widgets Disabled
        self.set_widgets_enable(False)
        # Initialize Image Label
        self.ui.label_image.setText("")
        self.ui.label_image.setFrameStyle(0)

    def button_run_click(self):
        """ Handle PROCESS button clicked. """
        if self.ui.checkBox_remove_objects.isChecked():
            self.__image.remove_object(self.ui.label_image.drop_mask)
        else:
            resized_width, resized_height = int(self.ui.edit_width.text()), int(self.ui.edit_height.text())
            if self.ui.checkBox_face.isChecked():
                for (x, y, w, h) in self.ui.label_image.face_result:
                    self.ui.label_image.keep_mask[y:y + h, x:x + w] = True
            self.__image.resize(resized_width, resized_height, self.ui.label_image.keep_mask)

        q_img = QImage(self.__image.img, self.__image.width(), self.__image.height(),
                       self.__image.width() * 3, QImage.Format_RGB888)
        self.ui.label_image.src_pixmap = QPixmap.fromImage(q_img)
        self.ui.label_image.setPixmap(self.ui.label_image.src_pixmap)
        self.ui.label_image.setFrameStyle(1)

        self.__update_edit_size_info()
        self.__set_checkbox_unchecked()
        self.ui.button_save_image.setEnabled(True)

    def button_open_image_click(self):
        """ Handle OPEN IMAGE button clicked. """
        img_name, img_type = QFileDialog.getOpenFileName(
            self, "Open Image", "./", "Images (*.jpg *.png *.bmp);;All Files(*)"
        )
        # save and show image
        if len(img_name) > 0:
            # record filename and path
            img_name_split_list = img_name.split('/')
            self.__image_name = img_name_split_list[-1]
            self.__image_path = '/'.join(img_name_split_list[:len(img_name_split_list) - 1])
            # read image
            img = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.__image.set_image(img)
            q_img = QImage(self.__image.img, self.__image.width(), self.__image.height(),
                           self.__image.width() * 3, QImage.Format_RGB888)
            self.ui.label_image.src_pixmap = QPixmap.fromImage(q_img)
            self.ui.label_image.setPixmap(self.ui.label_image.src_pixmap)
            self.ui.label_image.setFrameStyle(1)

            self.set_widgets_enable(True)
            self.__update_edit_size_info()
            self.__set_checkbox_unchecked()
            # Cannot Save Image
            self.ui.button_save_image.setDisabled(True)

    def button_save_image_click(self):
        """ Handle SAVE IMAGE button clicked. """
        save_path = os.path.join(self.__image_path, self.__image_name.split('.')[0])
        img_name, img_type = QFileDialog.getSaveFileName(
            self, "Save Image", save_path, "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)"
        )
        if len(img_name) > 0:
            q_img = QImage(self.__image.img, self.__image.width(), self.__image.height(),
                           self.__image.width() * 3, QImage.Format_RGB888)
            q_img.save(img_name)
            self.ui.button_save_image.setDisabled(True)

    def checkbox_face_state_changed(self):
        if not self.ui.checkBox_face.isChecked():
            if not self.ui.checkBox_keep_objects.isChecked() and not self.ui.checkBox_remove_objects.isChecked():
                self.ui.label_image.set_paint_mode(-1)
            # Clear Label
            self.ui.label_image.face_result = None
            self.ui.label_image.clear_mask()
        else:
            self.ui.label_image.set_paint_mode(2)
            if self.ui.checkBox_remove_objects.isChecked():
                self.ui.checkBox_remove_objects.setChecked(Qt.Unchecked)
            if self.ui.checkBox_keep_objects.isChecked():
                self.ui.checkBox_keep_objects.setChecked(Qt.Unchecked)
            # Recognize Faces
            face_patterns = cv2.CascadeClassifier('./src/haarcascade_frontalface_default.xml')
            self.ui.label_image.face_result = face_patterns.detectMultiScale(
                self.__image.img, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
            self.__image.keep_mask = np.zeros((self.__image.height(), self.__image.width()), dtype=bool)
            for (x, y, w, h) in self.ui.label_image.face_result:
                self.ui.label_image.keep_path.addRect(QRectF(x, y, w, h))
            # Refresh Label
            self.ui.label_image.paint_keep_mask()

    def checkbox_keep_objects_state_changed(self):
        if not self.ui.checkBox_keep_objects.isChecked():
            if not self.ui.checkBox_face.isChecked() and not self.ui.checkBox_remove_objects.isChecked():
                self.ui.label_image.set_paint_mode(-1)
        else:
            self.ui.label_image.set_paint_mode(1)
            if self.ui.checkBox_remove_objects.isChecked():
                self.ui.checkBox_remove_objects.setChecked(Qt.Unchecked)
            if self.ui.checkBox_face.isChecked():
                self.ui.checkBox_face.setChecked(Qt.Unchecked)
        self.ui.label_image.clear_mask()

    def checkbox_remove_objects_state_changed(self):
        if not self.ui.checkBox_remove_objects.isChecked():
            self.ui.edit_width.setEnabled(True)
            self.ui.edit_height.setEnabled(True)
            if not self.ui.checkBox_face.isChecked() and not self.ui.checkBox_keep_objects.isChecked():
                self.ui.label_image.set_paint_mode(-1)
        else:
            self.ui.edit_width.setDisabled(True)
            self.ui.edit_height.setDisabled(True)
            self.ui.label_image.set_paint_mode(0)
            if self.ui.checkBox_face.isChecked():
                self.ui.checkBox_face.setChecked(Qt.Unchecked)
            if self.ui.checkBox_keep_objects.isChecked():
                self.ui.checkBox_keep_objects.setChecked(Qt.Unchecked)
        self.ui.label_image.clear_mask()

    def set_widgets_enable(self, enabled: bool):
        """ Set Widgets of Application Window ENABLED or DISABLED. """
        if not enabled:
            # Disable Buttons
            self.ui.button_save_image.setDisabled(True)
            self.ui.button_run.setDisabled(True)
            # Disable Edits
            self.ui.edit_height.setDisabled(True)
            self.ui.edit_width.setDisabled(True)
            # Disable CheckBoxs
            self.ui.checkBox_face.setDisabled(True)
            self.ui.checkBox_remove_objects.setDisabled(True)
            self.ui.checkBox_keep_objects.setDisabled(True)
        else:
            # Enable Buttons
            self.ui.button_save_image.setEnabled(True)
            self.ui.button_run.setEnabled(True)
            # Enable Edits
            self.ui.edit_height.setEnabled(True)
            self.ui.edit_width.setEnabled(True)
            # Enable CheckBoxs
            self.ui.checkBox_face.setEnabled(True)
            self.ui.checkBox_remove_objects.setEnabled(True)
            self.ui.checkBox_keep_objects.setEnabled(True)

    def __update_edit_size_info(self):
        """ Update the SIZE INFORMATION of Edits. """
        self.ui.edit_width.setText(str(self.__image.width()))
        self.ui.edit_height.setText(str(self.__image.height()))

    def __set_checkbox_unchecked(self):
        self.ui.checkBox_face.setChecked(Qt.Unchecked)
        self.ui.checkBox_remove_objects.setChecked(Qt.Unchecked)
        self.ui.checkBox_keep_objects.setChecked(Qt.Unchecked)
