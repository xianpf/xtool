import torch
from PyQt5 import QtWidgets, QtCore
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QComboBox, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QGroupBox, QSpinBox
import numpy as np
import matplotlib.patches as patches
import cv2
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

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
        self.figure_right_gt_mask = plt.figure('right_gt_mask')

        self.ax_left = self.figure_left.add_subplot(111)
        self.ax_right = self.figure_right.add_subplot(111)
        self.ax_right_down = self.figure_right_gt_mask.add_subplot(111)

        self.canvas_left = FigureCanvas(self.figure_left)
        self.canvas_right = FigureCanvas(self.figure_right)
        self.canvas_right_down = FigureCanvas(self.figure_right_gt_mask)

        self.toolbar_right_down = NavigationToolbar(self.canvas_right_down, self)
        # self.toolbar = NavigationToolbar(self.canvas, self)

            # self.typeBox = QComboBox()
            # self.typeBox.addItem('line chart')
            # self.typeBox.addItem('scatter chart')

            # self.titleBox = QLineEdit("Graph", self)
            # self.xLabelBox = QLineEdit("x", self)
            # self.yLabelBox = QLineEdit("y", self)

            # self.xBox = QLineEdit("1,2,3,4,5", self)
            # self.yBox = QLineEdit("23,32,53,75,23", self)

            # self.btn_plot = QPushButton("plot", self)
            # self.btn_plot.clicked.connect(self.plot)

        self.channel_spin = QSpinBox(self)
        self.batch_inx_spin = QSpinBox(self)
        self.tuple_inx_spin = QSpinBox(self)
        self.shape_label = QLabel(self)
        self.channel_spin.valueChanged.connect(self.callback_tensor_update)
        self.batch_inx_spin.valueChanged.connect(self.callback_tensor_update)
        self.tuple_inx_spin.valueChanged.connect(self.callback_tensor_update)
        self.comment_text = 'Comment:'
        self.comment_label = QLabel(self)
        
        self.initUI()

    def initUI(self):
        self.setGeometry(1900, 0, 1610, 1000)
        spinsetting_layout = QHBoxLayout()
        spinsetting_layout.addWidget(QLabel("Channel"))
        spinsetting_layout.addWidget(self.channel_spin)
        spinsetting_layout.addWidget(QLabel("BatchInx"))
        spinsetting_layout.addWidget(self.batch_inx_spin)
        spinsetting_layout.addWidget(QLabel("TupleInx"))
        spinsetting_layout.addWidget(self.tuple_inx_spin)
        spinsetting_layout.addWidget(self.shape_label)
        spinsetting_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        spinsettingHBox = QWidget()
        spinsettingHBox.setLayout(spinsetting_layout)

        left_layout = QVBoxLayout()
        left_down_layout = QVBoxLayout()
        left_down_widget = QWidget()
        left_down_widget.setLayout(left_down_layout)
        # left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas_left)
        left_layout.addWidget(spinsettingHBox)
        left_layout.addWidget(self.comment_label)
        left_layout.addWidget(left_down_widget)
        left_layout.setStretch(0,5)
        left_layout.setStretch(1,1)
        left_layout.setStretch(2,1)
        left_layout.setStretch(3,3)
        # left_widget = QWidget()
        left_widget = QGroupBox("Left")
        left_widget.setLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas_right)
        right_layout.addWidget(self.canvas_right_down)
        # right_layout.addWidget(QLabel("type of Chart:"))
            # right_layout.addWidget(self.typeBox)
            # right_layout.addWidget(QLabel("title:"))
            # right_layout.addWidget(self.titleBox)
            # right_layout.addWidget(QLabel("x label:"))
            # right_layout.addWidget(self.xLabelBox)
            # right_layout.addWidget(QLabel("y label:"))
            # right_layout.addWidget(self.yLabelBox)

            # right_layout.addWidget(QLabel("x value:"))
            # right_layout.addWidget(self.xBox)
            # right_layout.addWidget(QLabel("y value:"))
            # right_layout.addWidget(self.yBox)
            # right_layout.addStretch(5)
            # right_layout.addWidget(self.btn_plot)
        
        right_down_layout = QVBoxLayout()
        right_down_layout.addWidget(self.toolbar_right_down)
        right_down_widget = QWidget()
        right_down_widget.setLayout(right_down_layout)
        right_layout.addWidget(right_down_widget)
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
    # def update_left(self, img):
        # self.ax_left.imshow(img)
        # self.canvas_left.draw()

    def update_right(self, img):
        '''from xtool.qt2image import win
        win.update_right((images.tensors.permute(0, 2, 3, 1).detach().cpu().numpy()[0]+128).astype(np.uint8))'''
        # import pdb; pdb.set_trace()
        # if len(img.shape) >= 3:
        #     img = img.transpose()
        self.right_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax_right.imshow(self.right_img)
        self.canvas_right.draw()

    def draw_rect_on_right(self, rect=None, clean='all'):
        '''clean: - 'all': 全都清除, 啥也不留
        - 'onlynew': 清除以前的, 保留最新的
        - 'keep': 保留以前的, 并在上面继续画'''
        if clean == 'all':
            self.ax_right.clear()
            self.ax_right.imshow(self.right_img)
        else:
            if rect is not None and len(rect) == 4:
                x1, y1, x2, y2 = rect
                rectpatch = patches.Rectangle((x1,y1),x2-x1,y2-y1,\
                    linewidth=1,edgecolor='r',facecolor='none')
                if clean == 'onlynew':
                    self.ax_right.clear()
                    self.ax_right.imshow(self.right_img)
                    self.ax_right.add_patch(rectpatch)
                elif clean == 'keep':
                    self.ax_right.add_patch(rectpatch)
                else:
                    print('clean只应该为 "all", "onlynew", "keep"三者之一')
            else:
                print('此处的rect不应该为None')
                import pdb; pdb.set_trace()
        self.canvas_right.draw()


    def update_right_down(self, img):
        '''from xtool.qt2image import win
        win.update_right((images.tensors.permute(0, 2, 3, 1).detach().cpu().numpy()[0]+128).astype(np.uint8))'''
        # import pdb; pdb.set_trace()
        # if len(img.shape) >= 3:
        #     img = img.transpose()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax_right_down.imshow(img)
        self.canvas_right_down.draw()

    def update_right_process(self, img, work_area):
        # import pdb; pdb.set_trace()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.add(img, np.zeros(np.shape(img), dtype=np.float32), mask=work_area)
        self.ax_right_down.imshow(img)
        self.canvas_right_down.draw()

    def update_left(self, data):
        '''input should in pattern: tuple(batch, channel, height, width)'''
        self.update_left_input_data = data
        self.channel_spin.setValue(0)
        self.batch_inx_spin.setValue(0)
        self.tuple_inx_spin.setValue(0)
        if isinstance(data, tuple):
            import pdb; pdb.set_trace()
        elif isinstance(data, np.ndarray):
            self.left_nparray_shape = data.shape
            self.shape_label.setText('  nparray:'+str(self.left_nparray_shape))
            if len(self.left_nparray_shape) == 2:
                self.left_img = data
                self.channel_spin.setDisabled(True)
                self.batch_inx_spin.setDisabled(True)
                self.tuple_inx_spin.setDisabled(True)
            elif len(self.left_nparray_shape) == 3:
                self.left_img = data[0]
                self.channel_spin.setDisabled(False)
                self.channel_spin.setMaximum(self.left_nparray_shape[0]-1)
                self.batch_inx_spin.setDisabled(True)
                self.tuple_inx_spin.setDisabled(True)
            elif len(self.left_nparray_shape) == 4:
                self.left_img = data[0][0]
                self.channel_spin.setDisabled(False)
                self.channel_spin.setMaximum(self.left_nparray_shape[1]-1)
                self.batch_inx_spin.setDisabled(False)
                self.batch_inx_spin.setMaximum(self.left_nparray_shape[0]-1)
                self.tuple_inx_spin.setDisabled(True)
            else:
                print('len(self.left_nparray_shape):', len(self.left_nparray_shape))
                import pdb; pdb.set_trace()
        # elif isinstance(data, torch.Tensor):
        #     self.update_left(data.detach().cpu().numpy())
        else:
            print('Type:', type(data))
            import pdb; pdb.set_trace()
        self.ax_left.imshow(self.left_img)
        self.canvas_left.draw()
        # import pdb; pdb.set_trace()

    def callback_tensor_update(self):
        if self.shape_label.text().startswith('  nparray:'):
            if len(self.left_nparray_shape) == 3:
                self.left_img = self.update_left_input_data[self.channel_spin.value()]
            elif len(self.left_nparray_shape) == 4:
                self.left_img = self.update_left_input_data[self.batch_inx_spin.value()][self.channel_spin.value()]
            self.ax_left.imshow(self.left_img)
            self.canvas_left.draw()
        else:
            import pdb; pdb.set_trace()

    def comment(self, text, mode='append'):
        self.comment_text = text if mode == 'refresh' else self.comment_text + text
        self.comment_label.setText(self.comment_text)


# if __name__ == '__main__':
#     if QtWidgets.QApplication.instance() is None:
#         app = QtWidgets.QApplication(sys.argv)
#     win = Window()
#     win.show()

if QtWidgets.QApplication.instance() is None:
    app = QtWidgets.QApplication(sys.argv)
win = Window()
win.show()