# version 0.1.0

from matplotlib import pyplot as plt
import torch, sys, cv2, skimage
import scipy.misc
import numpy as np
from threading import Thread
from matplotlib.widgets import Button, TextBox, Slider, CheckButtons

class XimgWeightShow():
    '''一般weight是个4维tensor(B,C,H,W), 这里只能输入(C,H,W)展示
        若chaneltype='rgb',则可按对应颜色展示
        mode 可以指定是show还是save
    '''
    def __init__(self, weights=None, kernel_size=3, channel_num=3, mode='show'):
        # plt.ion()
        # self.fig = plt.figure('XimgWeightShow', figsize=(6,2),facecolor='white')
        # self.fig.show()
        self.channel_num = channel_num
        self.channels = []
        self.n = kernel_size ** 2
        # self.meta_pos = np.array(np.meshgrid(np.linspace(0,1,kernel_size+2)[1:-1], 
        #     np.linspace(0,1,kernel_size+2)[1:-1])).transpose((1,2,0)).reshape(-1, 2)
        self.meta_pos = np.array(np.meshgrid(np.linspace(0.5/kernel_size,1-0.5/kernel_size,kernel_size), 
            np.linspace(0.5/kernel_size,1-0.5/kernel_size,kernel_size))).transpose((1,2,0)).reshape(-1, 2)
        if weights is not None:
            self.weights_np = weights.clone().detach().numpy()
            self.channel_num = len(self.weights_np)
            self.process()
                
    def __del__(self):
        plt.ioff()

    '''
    inputtype 可以是tensor, nparray, 或者list
    '''
    def updateWeights(self, weights, inputtype='tensor', show=True):
        if inputtype == 'tensor': 
            self.weights_np = weights.clone().detach().cpu().numpy()
        elif inputtype == 'nparray': 
            self.weights_np = weights
        elif inputtype == 'list': 
            # import pdb; pdb.set_trace()
            self.weights_np = np.array(weights)
        self.channel_num = len(self.weights_np)
        # self.process()
        return self.process(scope='crossChannels', show=show)

    def process(self, scope='perChannel', show=True):
        self.channels.clear()
        self.w_num = 8 if self.channel_num >= 8 else self.channel_num
        self.h_num = (self.channel_num-1)//8+1 if self.channel_num >= 8 else 1
        # import pdb; pdb.set_trace()
        if show:
            plt.ion()
        # plt.rcParams["figure.figsize"] = (6,2)
        # self.fig.set_size_inches(self.w_num*2,self.h_num*2)
        self.fig = plt.figure('XimgWeightShow')
        self.fig.set_size_inches(self.w_num,self.h_num)
        plt.clf()
        self.total_sum_abs = np.abs(self.weights_np).sum()
        for index, data in enumerate(self.weights_np):
            channel = dict()
            channel['data'] = data
            channel['sum'] = data.sum()
            channel['abs_sum'] = np.abs(data).sum()
            channel['abs_data'] = np.abs(data)
            channel['signs'] = (data > 0).reshape(-1).astype(np.float)
            if scope=='perChannel':
                # channel['sizes'] = np.abs(data.reshape(-1)*200/data.sum())
                channel['sizes'] = channel['abs_data'].reshape(-1)*2000/channel['abs_sum']
            elif scope=='crossChannels':
                channel['sizes'] = channel['abs_data'].reshape(-1)*2000*self.channel_num/self.total_sum_abs
            color = np.ones((self.n, 4))                     #index  0   1   2 
            if self.channel_num == 3:
                color[:,index] = channel['signs']            #       0   1   2
                color[:,(index+1)%3] = 1 - channel['signs']  #       1   2   0
                color[:,(index+2)%3] = 1 - channel['signs']  #       2   0   1
                channel['colors'] = color
            else:
                channel['colors'] = ['#FF6347' if sign else '#1E90FF' for sign in channel['signs']]

            self.channels.append(channel)
            # import pdb; pdb.set_trace()
            plt.subplot(self.h_num, self.w_num, index+1)
            plt.xlim(0,1), plt.xticks([]), plt.ylim(0,1), plt.yticks([])
            plt.scatter(self.meta_pos[:,0], self.meta_pos[:,1], 
                    s=self.channels[index]['sizes'], lw = 0.5, 
                    edgecolors = self.channels[index]['colors'], facecolors=self.channels[index]['colors'])
            
            # plt.text(0.05,0.05,'as:'+str(round(self.channels[index]['abs_sum'], 4))+';1:'+\
            #     str(round(self.channels[index]['data'][0][0], 4)))
            plt.text(0.05,0.05,'as:'+str(round(self.channels[index]['abs_sum'], 4)))
        # import pdb; pdb.set_trace()
        plt.tight_layout(pad=0.5)
        # plt.pause(0.1)
        self.fig.canvas.flush_events()
        plt.draw()
        # plt.pause(0.2)
        plt.savefig(sys.path[0]+'/tmptmp.png')
        plt.ioff()

        return sys.path[0]+'/tmptmp.png'
# wshow = XimgWeightShow()

class DynamicShow():
    def __init__(self, window_name, wait=1, win_size=None, fullscreen=False):
        """
        :param wait: wait time (ms) after showing image, minimum: 1
        :param win_size: the size of the window created
        :param fullscreen: boolean if scale to fullscreen
        """
        self.window_name = window_name
        self.frame_wait = wait
        self.curr_frame = None
        self.started_thread = False
        if fullscreen:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if win_size is not None:
            cv2.resizeWindow(window_name, win_size[0], win_size[1])

    def __del__(self):
        try:
            self.t._Thread__stop()
            cv2.destroyWindow(self.window_name)
        except:
            pass

    # fig.canvas.draw()
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
    #     sep='')
    # img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    def show(self, frame):
        """
        :param frame:
        :return: True if keep playing, False if stop
        # TODO: do by try exception
        """
        self.curr_frame = frame
        # if not self.started_thread:
        #     self.start()
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(self.frame_wait) & 0xFF == ord('q'):
        # if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyWindow(self.window_name)
            return False
        return True
    
    def start(self):
        t = Thread(target=self.update, args=())
        self.t = t
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            # if the thread indicator variable is set, stop the thread
            # if self.stopped:
            #     self.t._Thread__stop()
            #     return
            cv2.imshow(self.window_name, self.curr_frame)
            if cv2.waitKey(self.frame_wait) & 0xFF == ord('q'):
                self.t._Thread__stop()
                cv2.destroyWindow(self.window_name)
                return False

class PlotShow():
    def __init__(self, window_name, frame_size=None, fullscreen=False):
        self.frame_size = frame_size
        # self.fig = plt.figure(window_name, figsize=(16,9))
        plt.ion()
        # fig, ax = plt.subplots()  
        self.fig, self.ax = plt.subplots()  
        self.fig.canvas.set_window_title('Qt3.0-show.exe.AppImage')
        self.fig.set_size_inches(18,10)
        mgr = plt.get_current_fig_manager()
        py = mgr.canvas.height()
        px = mgr.canvas.width()
        mgr.window.setGeometry(1900, 0, px, py)
        # plt.get_current_fig_manager().full_screen_toggle()
        # plt.subplots_adjust(bottom=0.2,left=0.3) #调整子图间距
        if not frame_size:
            frame_size = (6,8)
        init_img = np.zeros(frame_size+(3,))
        self.img_ax =plt.imshow(init_img)
        # self.fig.canvas.flush_events()
        plt.draw()
        plt.pause(0.2)
        # plt.show()
        
    def __del__(self):
        plt.ioff()

    def update(self, newImage, resize=True):
        if resize:
            newImage = cv2.resize(newImage, self.frame_size, interpolation=cv2.INTER_CUBIC)
            newImage = np.clip(newImage, 0.0, 1.0)
        self.img_ax.set_data(newImage)

def channel2grayRGB(channel_img, datatype='highconctract'):
    if datatype == 'int8':
        channel_img = (channel_img+128)/255.0
    elif datatype == 'highconctract':
        cmax, cmin = channel_img.max(), channel_img.min()
        if cmax == cmin:
            channel_img = channel_img if cmax <= 1 and cmin >= 0 else 0
        else:
            channel_img = (channel_img - cmin) / (cmax - cmin)
    grayR = grayG = grayB = channel_img
    grayRGB = np.stack([grayR, grayG, grayB], axis=-1)
    return grayRGB

def relative_ax(parent_ax, dx=0, dy=0, dw=0, dh=0, abs_size=False):
    '''abs_size 为真时,dw和dh取绝对大小而不是相对大小
    '''
    parent_x, parent_y, parent_w, parent_h = list(parent_ax._position._points.reshape(-1))
    parent_w = parent_w-parent_x
    parent_h = parent_h-parent_y
    x = parent_x + dx
    y = parent_y + dy
    w = dw if abs_size else parent_w+dw
    h = dh if abs_size else parent_h+dh
    return x, y, w, h

class FeatureShowiON(PlotShow):
    def __init__(self, window_name, frame_size):
        super(FeatureShow, self).__init__(window_name, frame_size)
        self.img_ax.axes.set_position([-0.09, 0.36, 0.6, 0.6])
        axfpos = plt.axes([0.05, 0.3, 0.3, 0.025]) #从左下角的位置 x, y, w, h 占ax的百分比
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.tfpos = TextBox(axfpos, 'Curr Pos', initial='none')
    
        def prev_callback(ss):
            self.tfpos.set_val('node.namexxxpppppre')
        def next_callback(ss):
            self.tfpos.set_val('nnnnnnnnnnnnnnnn')
        bnext = Button(axnext, 'NextLayer')
        bnext.on_clicked(next_callback)
        bprev = Button(axprev, 'PrevLayer')
        # bprev.on_clicked(prev_callback)

    def update(self, newImage, node, resize=True):
        if resize:
            newImage = cv2.resize(newImage, self.frame_size, interpolation=cv2.INTER_CUBIC)
            newImage = np.clip(newImage, 0.0, 1.0)
        self.img_ax.set_data(newImage)
        self.tfpos.set_val(node.name)

class FeatureShow():
    def __init__(self, window_name, frame_size, rootNode):
        self.frame_size = frame_size
        self.rootNode = rootNode
        self.curr_understand_node(list(self.rootNode.children_nodes.values())[0])
        self.index_for_list_feature = 0


        self.fig, self.ax = plt.subplots()  
        self.fig.canvas.set_window_title('Qt3.0-show.exe.AppImage')
        mgr = plt.get_current_fig_manager()
        py = mgr.canvas.height()
        px = mgr.canvas.width()
        mgr.window.setGeometry(1900, 0, px, py)
        if not frame_size:
            frame_size = (6,8)
        self.fig.set_size_inches(18,10)
        self.init_img = np.zeros(frame_size+(3,))
        self.img_ax =plt.imshow(self.init_img)
        self.img_ax.axes.set_position([-0.09, 0.36, 0.6, 0.6])
        axfpos = plt.axes([0.05, 0.3, 0.3, 0.025]) #从左下角的位置 x, y, w, h 占ax的百分比
        self.tfpos = TextBox(axfpos, 'Curr Pos', initial='none')
        axstat = plt.axes([0.4, 0.36, 0.1, 0.6])
        self.tstat = TextBox(axstat, '', initial='none\nnone\nnone')
        
    
        # self.update_widgets()
        
        def prev_callback(event):
            if self.curr_node.parent_node.name != 'root':
                self.curr_node.parent_node.child_index = self.curr_node.sibling_index
                self.curr_understand_node(self.curr_node.parent_node)
                self.update_widgets()
        def next_callback(event):
            if len(self.curr_node.children_nodes) > 0:
                self.curr_understand_node(list(\
                    self.curr_node.children_nodes.values())[self.curr_node.child_index])
                self.update_widgets()
        def up_callback(event):
            if self.curr_node.sibling_index > 0:
                self.curr_understand_node(list(\
                    self.curr_node.parent_node.children_nodes.values())[self.curr_node.sibling_index-1])
                self.update_widgets()
        def down_callback(event):
            if self.curr_node.sibling_index < len(self.curr_node.parent_node.children_nodes)-1:
                self.curr_understand_node(list(\
                    self.curr_node.parent_node.children_nodes.values())[self.curr_node.sibling_index+1])
                self.update_widgets()
        def prev_channel_callback(event):
            if self.curr_node.channel_index > 0:
                self.curr_node.channel_index -= 1
                self.update_widgets()
        def next_channel_callback(event):
            if self.curr_node.channel_index < self.curr_node.channel_num-1:
                self.curr_node.channel_index += 1
                self.update_widgets()
        def channelS_callback(val):
            print('xxxxxxxxx', val, int(round(val)))
            self.curr_node.channel_index = int(round(val))


        axprev = plt.axes([0.04, 0.25, 0.05, 0.025])
        axnext = plt.axes(relative_ax(axprev, dx=0.06))
        axup = plt.axes(relative_ax(axnext, dx=0.06))
        axdown = plt.axes(relative_ax(axup, dx=0.06))
        axcprev = plt.axes(relative_ax(axdown, dx=0.06))
        axcnext = plt.axes(relative_ax(axcprev, dx=0.06))
        # axchannelSider = plt.axes(relative_ax(axprev, dy=-0.03, dw=0.35))
        axstatcheck = plt.axes(relative_ax(axprev, dy=-0.03))

        bnext = Button(axnext, 'NextLayer')
        bnext.on_clicked(next_callback)
        bprev = Button(axprev, 'PrevLayer')
        bprev.on_clicked(prev_callback)
        bup = Button(axup, 'UpNode')
        bup.on_clicked(up_callback)
        bdown = Button(axdown, 'DownNode')
        bdown.on_clicked(down_callback)
        bcprev = Button(axcprev, 'PrevChnl')
        bcprev.on_clicked(prev_channel_callback)
        bcnext = Button(axcnext, 'NextChnl')
        bcnext.on_clicked(next_channel_callback)
        # self.schannel = Slider(axchannelSider, 'channel', valmin=0, valmax=2, valinit=0, valfmt='%d')
        # self.schannel.on_changed(channelS_callback)

        self.update_widgets()
        plt.show()


    def update_widgets(self, resize=False):
        curr_channel_feature_map = self.feature_map[self.curr_node.channel_index]
        newImage = channel2grayRGB(curr_channel_feature_map)
        if resize:
            newImage = cv2.resize(newImage, self.frame_size, interpolation=cv2.INTER_CUBIC)
            newImage = np.clip(newImage, 0.0, 1.0)
        self.img_ax.set_data(newImage)
        # self.schannel.set_val(self.curr_node.channel_index)
        # self.schannel.valmax = self.curr_node.channel_num-1
        self.tfpos.set_val(self.curr_node.name+'    '+\
            str(self.curr_node.sibling_index)+' of '+\
            str(self.curr_node.sibling_num)+'    chn:'+\
            str(self.curr_node.channel_index)+' of '+\
            str(self.curr_node.channel_num))
        self.tstat.set_val(
            'name: root.'+ str(self.curr_node.name) + '\n' + \
            'shape:'+ str(self.feature_map.shape)+'\n'+\
            'sibling: '+ str(self.curr_node.sibling_index)+' of '+\
                str(self.curr_node.sibling_num) + '\n' + \
            'channel: '+ str(self.curr_node.channel_index)+' of '+\
                str(self.curr_node.channel_num) + '\n' + \
            'max: '+ str(curr_channel_feature_map.max()) + '\n' + \
            'min: '+ str(curr_channel_feature_map.min()) + '\n' + \
            'mean: '+ str(curr_channel_feature_map.mean()) + '\n' + \
            'sum: '+ str(curr_channel_feature_map.sum()) + '\n' + \
            'std_var: '+ str(curr_channel_feature_map.std()) + '\n' + \
            'variance: '+ str(curr_channel_feature_map.var()) + '\n' + \
            'len_net: '+ str(len(self.network_code)) + '\n' + \
            'len_forward: '+ str(len(self.forward_code)) + '\n' + \
            'Your Majesty.'\
            + ('\n\n' + self.parent_net if len(self.parent_net)<1000 else '\n...') \
            + ('\n\n' + self.network_code if len(self.network_code)<1000 else '\n...') \
            + '\n\n' + self.forward_code \
        )
        # self.tstat.cursor._visible = False


    def curr_understand_node(self, node):
        self.curr_node = node
        print('key:', node.feature_buf_key)
        forward_codes = self.rootNode.forward_codes[node.feature_buf_key]
        self.network_code, self.forward_code = forward_codes.split('@')
        if node.parent_node.name == 'root':
            self.parent_net = 'root'
        else:
            self.parent_net = self.rootNode.forward_codes[node.parent_node.feature_buf_key].split('@')[0]
        feature_map = self.rootNode.feature_buf[node.feature_buf_key]
        feature_map = feature_map if isinstance(feature_map, np.ndarray) else feature_map[self.index_for_list_feature]
        self.feature_map = feature_map[0]
        self.curr_node.sibling_num = len(self.curr_node.parent_node.children_nodes)
        self.curr_node.channel_num = len(self.feature_map)
        # print(node.name, node.sibling_index, len(node.parent_node.children_nodes))
        # print(type(feature_map))
        # print(type(feature_map), feature_map.shape)

