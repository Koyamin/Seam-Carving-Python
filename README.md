# Seam-Carving-Python

A Seam-Carving Application with GUI by using Python 3. The GUI is builded by PyQT5.

## Features of Project

This project has these features:
- Automatically resize the size of an image.
- Keep face when resizing an image if exists.
- Select and Erase an area by using mouse.

## Details of Project

### Development Environment

- Operating System: Windows 10 21H2
- Code Language: Python 3.9
- IDE: Pycharm 2022.1 (Professional Edition)

### Package Requirements

Packages this project required are as followes.
```
click==7.1.2
numpy==1.22.3
opencv-python==4.5.5.64
PyQt5==5.15.4
pyqt5-plugins==5.15.4.2.2
PyQt5-Qt5==5.15.2
PyQt5-sip==12.10.1
pyqt5-tools==5.15.4.3.2
python-dotenv==0.20.0
qt5-applications==5.15.2.2.2
qt5-tools==5.15.2.1.2
scipy==1.8.0
```
All the packages required are written in `requirements.txt`.

## Project Structure

The structure of this project is: 
```
├─ src                                      // Source file
    ├─ icons                                // Icons
        ├─ ...
    ├─ haarcascade_frontalface_default.xml  // Face Detector
    ├─ ...
├─ CustomizedWidgets.py                     // Widgets
├─ imageData.py                             // Model Class
├─ img_src.qrc                              // PyQt Resources File
├─ img_src_rc.py                            // PyQt Resources File
├─ main.py                                  // Main Program
├─ MainWindow.py                            // PyQt GUI File
├─ MainWindow.ui                            // PyQt GUI File
├─ requirements.txt                         // Packege Requirements Document
├─ seam_carving.py                          // Seam-Carving
└─ UiController.py                          // UI Controller Class
```

