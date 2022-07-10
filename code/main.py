import sys
from PyQt5.QtWidgets import QApplication
from UiController import MyWindow


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        my_window = MyWindow()
        my_window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(e)
