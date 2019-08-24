from PyQt5 import QtWidgets
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QComboBox, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QGroupBox
import numpy as np


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # set the title of main window
        self.setWindowTitle('Plot tool')

        # # set the size of window
        # self.Width = 800
        # self.height = int(0.618 * self.Width)
        # self.resize(self.Width, self.height)

        # create widgets
        self.figure_left = plt.figure('left')
        self.figure_right = plt.figure('right')

        self.ax_left = self.figure_left.add_subplot(111)
        self.ax_right = self.figure_right.add_subplot(111)

        self.canvas_left = FigureCanvas(self.figure_left)
        self.canvas_right = FigureCanvas(self.figure_right)

        # self.toolbar = NavigationToolbar(self.canvas, self)

        self.typeBox = QComboBox()
        self.typeBox.addItem('line chart')
        self.typeBox.addItem('scatter chart')

        self.titleBox = QLineEdit("Graph", self)
        self.xLabelBox = QLineEdit("x", self)
        self.yLabelBox = QLineEdit("y", self)

        self.xBox = QLineEdit("1,2,3,4,5", self)
        self.yBox = QLineEdit("23,32,53,75,23", self)

        self.btn_plot = QPushButton("plot", self)
        self.btn_plot.clicked.connect(self.plot)
        
        self.initUI()

    def initUI(self):
        self.setGeometry(1900, 0, 1610, 1000)
        left_layout = QVBoxLayout()
        left_down_layout = QVBoxLayout()
        left_down_widget = QWidget()
        left_down_widget.setLayout(left_down_layout)
        # left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas_left)
        left_layout.addWidget(left_down_widget)
        # left_widget = QWidget()
        left_widget = QGroupBox("Left")
        left_widget.setLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas_right)
        right_layout.addWidget(QLabel("type of Chart:"))
        right_layout.addWidget(self.typeBox)
        right_layout.addWidget(QLabel("title:"))
        right_layout.addWidget(self.titleBox)
        right_layout.addWidget(QLabel("x label:"))
        right_layout.addWidget(self.xLabelBox)
        right_layout.addWidget(QLabel("y label:"))
        right_layout.addWidget(self.yLabelBox)

        right_layout.addWidget(QLabel("x value:"))
        right_layout.addWidget(self.xBox)
        right_layout.addWidget(QLabel("y value:"))
        right_layout.addWidget(self.yBox)
        right_layout.addStretch(5)
        right_layout.addWidget(self.btn_plot)
        right_widget = QGroupBox("Right")
        right_widget.setLayout(right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        # main_layout.setStretch(0, 4)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def plot(self):
        g_type = str(self.typeBox.currentText())

        x_data = str(self.xBox.text())
        y_data = str(self.yBox.text())

        x = np.array(x_data.split(",")).astype(np.float)
        y = np.array(y_data.split(",")).astype(np.float)

        if len(x) != len(y):
            QMessageBox.question(self, 'Message', "The size of X and Y is different. ", QMessageBox.Ok, QMessageBox.Ok)
        else:
            # ax = self.figure.add_subplot(111)
            # ax = self.figure_right.add_subplot(111)
            self.ax_right.clear()
            self.ax_right.set(xlabel=str(self.xLabelBox.text()), ylabel=str(self.yLabelBox.text()))
            self.ax_right.set(title=str(self.titleBox.text()))

            if g_type == 'line chart':
                self.ax_right.plot(x, y)
            elif g_type == 'scatter chart':
                self.ax_right.scatter(x, y)
            else:
                print("error.")

            # self.canvas.draw()
            self.canvas_right.draw()



# if __name__ == '__main__':
#     if QtWidgets.QApplication.instance() is None:
#         app = QtWidgets.QApplication(sys.argv)
#     win = Window()
#     win.show()
#     import pdb; pdb.set_trace()
#     print('hjhhhhhhhhhhhhhhhh')

if QtWidgets.QApplication.instance() is None:
    app = QtWidgets.QApplication(sys.argv)
win = Window()
win.show()
import pdb; pdb.set_trace()
print('hjhhhhhhhhhhhhhhhh')
