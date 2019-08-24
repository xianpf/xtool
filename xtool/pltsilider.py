from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
# import numpy as np
#
# np.set_printoptions(threshold=np.inf)

# img = io.imread("train.jpg")


# segments = slic(img, n_segments=1000, compactness=10)
# out=mark_boundaries(img,segments)
# # print(segments)
# plt.subplot(121)
# plt.title("n_segments=1000")
# plt.imshow(out)
# # plt.imshow(segments)

# segments2 = slic(img, n_segments=300, compactness=10)
# out2=mark_boundaries(img,segments2)
# plt.subplot(122)
# plt.title("n_segments=300")
# plt.imshow(out2)

# plt.tight_layout()
# plt.show()

from matplotlib.widgets import Slider,RadioButtons

def spv(omega1,omega2):
    x=np.linspace(0,20*np.pi,200)
    y1=np.sin(omega1*x)
    y2=np.sin(omega2*x)
    return (x,y1+y2)

fig, ax = plt.subplots()  
plt.subplots_adjust(bottom=0.2,left=0.3) #调整子图间距

x,y=spv(2,3)                             # 初始化函数
l,=plt.plot(x,y,color='red')             # 画出该条曲线

axcolor = 'lightgoldenrodyellow'  # slider的颜色
om1= plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor) # 第一slider的位置
om2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor) # 第二个slider的位置

som1 = Slider(om1, r'$\omega_1$', 1, 30.0, valinit=3) # 产生第一slider
som2 = Slider(om2, r'$\omega_2$', 1, 30.0, valinit=5) # 产生第二slider

def update(val):
    s1 = som1.val
    s2 = som2.val
    x,y=spv(s1,s2)
    l.set_ydata(y)
    l.set_xdata(x)
    fig.canvas.draw_idle()
som1.on_changed(update)
som2.on_changed(update)

cc = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(cc, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()