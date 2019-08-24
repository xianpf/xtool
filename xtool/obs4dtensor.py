import sys
from PyQt5.QtWidgets import QMainWindow, QComboBox, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QGroupBox, QSpinBox
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import torch


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # set the title of main window
        self.setWindowTitle('Plot tool')
        self.kernel_area_rect_patch = None


        # create widgets
        self.figure_left = plt.figure('left', constrained_layout={'w_pad': 0, 'h_pad': 0, 'wspace': 0, 'hspace': 0})
        self.figure_right = plt.figure('right', constrained_layout={'w_pad': 0, 'h_pad': 0, 'wspace': 0, 'hspace': 0})
        self.figure_right_gt_mask = plt.figure('right_gt_mask', constrained_layout={'w_pad': 0, 'h_pad': 0, 'wspace': 0, 'hspace': 0})

        self.ax_left = self.figure_left.add_subplot(111)
        self.ax_right = self.figure_right.add_subplot(111)
        self.ax_right_down = self.figure_right_gt_mask.add_subplot(111)

        self.canvas_left = FigureCanvas(self.figure_left)
        self.canvas_left.setFocusPolicy(QtCore.Qt.ClickFocus )
        self.canvas_left.setFocus()
        self.canvas_right = FigureCanvas(self.figure_right)
        self.canvas_right_down = FigureCanvas(self.figure_right_gt_mask)
        self.parentcanvas = self.ax_left.figure.canvas
 
        self.toolbar_left = NavigationToolbar(self.canvas_left, self)
        self.toolbar_right = NavigationToolbar(self.canvas_right, self)
        self.toolbar_right_down = NavigationToolbar(self.canvas_right_down, self)

        # self.kernelhalflength_spin = QSpinBox(self)
        # self.kernelhalflength_spin.setValue(50)
        self.channel_spin = QSpinBox(self)
        self.batch_inx_spin = QSpinBox(self)
        self.color_btn = QPushButton('Color')
        self.use_color_img = False
        # self.tuple_inx_spin = QSpinBox(self)
        self.shape_label = QLabel(self)
        self.channel_spin.valueChanged.connect(self.callback_tensor_update)
        self.batch_inx_spin.valueChanged.connect(self.callback_tensor_update)
        self.color_btn.clicked.connect(self.callback_color)
        self.comment_text = 'Comment:'
        self.comment_label = QLabel(self)
        # self.canvas_left.mousePressEvent = self.canvas_left_mouse
        
        self.initUI()

    def initUI(self):
        self.setGeometry(1900, 0, 1610, 1000)
        spinsetting_layout = QHBoxLayout()
        spinsetting_layout.addWidget(self.color_btn)
        spinsetting_layout.addWidget(QLabel("BatchInx"))
        spinsetting_layout.addWidget(self.batch_inx_spin)
        spinsetting_layout.addWidget(QLabel("Channel"))
        spinsetting_layout.addWidget(self.channel_spin)
        spinsetting_layout.addWidget(self.shape_label)
        spinsetting_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        spinsettingHBox = QWidget()
        spinsettingHBox.setLayout(spinsetting_layout)

        left_layout = QVBoxLayout()
        left_down_layout = QVBoxLayout()
        left_down_widget = QWidget()
        left_down_widget.setLayout(left_down_layout)
        left_layout.addWidget(self.toolbar_left)
        left_layout.addWidget(self.canvas_left)
        left_layout.addWidget(spinsettingHBox)
        left_layout.addWidget(self.comment_label)
        left_layout.addWidget(left_down_widget)
        left_layout.setStretch(0,1)
        left_layout.setStretch(1,5)
        left_layout.setStretch(2,1)
        left_layout.setStretch(3,1)
        left_layout.setStretch(4,2)
        # left_widget = QWidget()
        left_widget = QGroupBox("Left")
        left_widget.setLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.toolbar_right)
        right_layout.addWidget(self.canvas_right)
        right_down_layout = QVBoxLayout()
        right_layout.addWidget(self.toolbar_right_down)
        right_layout.addWidget(self.canvas_right_down)
        right_widget = QGroupBox("Right")
        right_widget.setLayout(right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def callback_color(self):
        self.use_color_img = not self.use_color_img
        if self.use_color_img and self.tensor4d.shape[1]==3:
            self.left_img = self.tensor4d[self.batch_inx_spin.value()].permute(1,2,0)
            self.left_img = (self.left_img - self.left_img.min())/(self.left_img.max() - self.left_img.min())
        else:
            self.left_img = self.tensor4d[self.batch_inx_spin.value()][self.channel_spin.value()]
        self.update_any_img(self.left_img, ax='left')


    def callback_tensor_update(self):
        self.left_img = self.tensor4d[self.batch_inx_spin.value()][self.channel_spin.value()]
        self.update_data()

    def set4dtensor(self, tensor):
        assert isinstance(tensor, torch.Tensor) and len(tensor.shape)==4, "只能输入4D的torch Tensor"
        self.tensor4d = tensor.detach().cpu()
        self.batch_inx_spin.setMaximum(self.tensor4d.shape[0]-1)
        self.channel_spin.setMaximum(self.tensor4d.shape[1]-1)
        self.shape_label.setText('Tensor Shape:'+str(self.tensor4d.shape))
        self.left_img = self.tensor4d[self.batch_inx_spin.value()][self.channel_spin.value()]
        self.update_data()

    def update_data(self):
        self.update_any_img(self.left_img, ax='left')
        # nnn, bins, patches = plt.hist(self.left_img.view(-1))
        self.ax_right.clear()
        self.ax_right.hist(self.left_img.contiguous().view(-1), bins=32)
        self.canvas_right.draw()
        # import pdb; pdb.set_trace()

    
    def update_any_img(self, img, ax):
        if ax=='left':
            self.ax_left.imshow(img)
            self.canvas_left.draw()
        elif ax=='right':
            self.ax_right.imshow(img)
            self.canvas_right.draw()
        elif ax=='right_down':
            self.ax_right_down.imshow(img)
            self.canvas_right_down.draw()

def startobs():
    if QtWidgets.QApplication.instance() is None:
        app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    return win

if QtWidgets.QApplication.instance() is None:
    app = QtWidgets.QApplication(sys.argv)
win4d = Window()
win4d.show()
# img = plt.imread('train.jpg')
# import pdb; pdb.set_trace()
# img_tensor = torch.Tensor(img).permute(2, 0, 1)[None,...]
# win4d.set4dtensor(img_tensor)