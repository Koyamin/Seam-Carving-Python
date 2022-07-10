from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import QMouseEvent, QPaintEvent, QPainter, QPainterPath, QPixmap, QColor
from PyQt5.QtWidgets import QLabel

import numpy as np


class DrawLabel(QLabel):
    def __init__(self, flags):
        super().__init__(flags)
        self.src_pixmap = None
        self.__current_point = None
        self.__paint_enabled = False
        self.paint_mode = -1
        self.__is_pressed = False
        self.drop_mask, self.drop_path = None, None
        self.keep_mask, self.keep_path = None, None
        self.face_result = None

    def set_paint_mode(self, mode: int):
        """
        Set Paint Mode of Mask.
        :param mode: Paint Mode of Mask. 0=drop mask; 1=keep mask.
        """
        self.paint_mode = mode
        if self.paint_mode == 0:
            self.__paint_enabled = True
            self.drop_mask = np.zeros((self.pixmap().height(), self.pixmap().width()), dtype=bool)
            self.drop_path = QPainterPath()
            self.drop_path.setFillRule(Qt.WindingFill)
            self.keep_mask, self.keep_path = None, None
        elif self.paint_mode == 1:
            self.__paint_enabled = True
            self.keep_mask = np.zeros((self.pixmap().height(), self.pixmap().width()), dtype=bool)
            self.keep_path = QPainterPath()
            self.keep_path.setFillRule(Qt.WindingFill)
            self.drop_mask, self.drop_path = None, None
        elif self.paint_mode == 2:
            self.__paint_enabled = False
            self.keep_mask = np.zeros((self.pixmap().height(), self.pixmap().width()), dtype=bool)
            self.keep_path = QPainterPath()
            self.keep_path.setFillRule(Qt.WindingFill)
            self.drop_mask, self.drop_path = None, None
        else:
            self.__paint_enabled = False
            self.drop_mask, self.drop_path = None, None
            self.keep_mask, self.keep_path = None, None
            # painter = QPainter(self.pixmap())
            # painter.drawPixmap(0, 0, self.src_pixmap)

    def paintEvent(self, e: QPaintEvent) -> None:
        super().paintEvent(e)
        if self.__is_pressed and self.__paint_enabled:
            x, y, w, h = self.__current_point.x() - 10, self.__current_point.y() - 10, 20, 20
            painter = QPainter(self.pixmap())
            painter.drawPixmap(0, 0, self.src_pixmap)
            if self.paint_mode == 0:
                self.drop_path.addRect(QRectF(x, y, w, h))
                painter.setPen(QColor(255, 0, 0, 100))
                painter.setBrush(QColor(255, 0, 0, 30))
                painter.drawPath(self.drop_path.simplified())
                self.drop_mask[max(y, 0):min(y + h, self.size().height()), max(x, 0):min(x + w, self.size().width())] = True
            elif self.paint_mode == 1:
                self.keep_path.addRect(QRectF(x, y, w, h))
                painter.setPen(QColor(0, 255, 0, 100))
                painter.setBrush(QColor(0, 255, 0, 30))
                painter.drawPath(self.keep_path.simplified())
                self.keep_mask[max(y, 0):min(y + h, self.size().height()), max(x, 0):min(x + w, self.size().width())] = True

    def paint_keep_mask(self):
        """ Paint Keep Mask. """
        painter = QPainter(self.pixmap())
        painter.drawPixmap(0, 0, self.src_pixmap)
        painter.setPen(QColor(0, 255, 0, 100))
        painter.setBrush(QColor(0, 255, 0, 30))
        painter.drawPath(self.keep_path)
        self.update()

    def clear_mask(self):
        """ Clear Painted Mask. """
        painter = QPainter(self.pixmap())
        painter.drawPixmap(0, 0, self.src_pixmap)
        self.update()

    def mousePressEvent(self, ev: QMouseEvent):
        super().mousePressEvent(ev)
        if ev.button() == Qt.LeftButton and self.__paint_enabled:
            self.__is_pressed = True
            self.__current_point = ev.pos()
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent):
        super().mouseMoveEvent(ev)
        if ev.buttons() & Qt.LeftButton and self.__paint_enabled:
            self.__current_point = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        super().mouseReleaseEvent(ev)
        if ev.button() == Qt.LeftButton and self.__paint_enabled:
            self.__is_pressed = False
            self.__current_point = ev.pos()
            self.update()
