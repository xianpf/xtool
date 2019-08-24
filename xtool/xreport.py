from matplotlib import pyplot as plt
import torch
import json, os, sys, shutil, imageio
import time, atexit
import numpy as np
from tinydb import TinyDB, Query, where
from PIL import Image, ImageDraw
# import xtool.visualize


def create_gif(image_list, gif_name):
 
    # frames = []
    # for image_name in image_list:
    #     frames.append(imageio.imread(baseurl+'/'+image_name))
    frames = [imageio.imread(image_name) for image_name in image_list]
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.5)


class Xreport():
    '''一般
    '''
    def __init__(self, exp_name='', base_path = '/home/xianr/data/trainlogs'):
        self.record_buffer = {}
        self.base_path = base_path
        self.curr_prj_path = sys.path[0]
        self.exp_name = exp_name
        self.init_time = time.time()
        self.init_time_str = time.strftime('%m%d_%H%M%S')
        self.folder_name = self.init_time_str+'Xrp_'+self.exp_name
        self.folder_path = self.base_path+'/'+self.folder_name
        if not os.path.exists(self.base_path+'/'+self.curr_prj_path.split('/')[-1]+'_saved'):
            os.makedirs(self.base_path+'/'+self.curr_prj_path.split('/')[-1]+'_saved')
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.curr_prj_log_path = self.curr_prj_path+'/log_xreport'
        if not os.path.exists(self.curr_prj_log_path):
            os.makedirs(self.curr_prj_log_path)
        if not os.path.exists(self.curr_prj_log_path+'/saved'):
            os.symlink(self.base_path+'/'+self.curr_prj_path.split('/')[-1]+'_saved', self.curr_prj_log_path+'/saved')
        os.symlink(self.folder_path, self.curr_prj_log_path+'/'+self.folder_name)
        self.file_name = self.init_time_str+'_'+self.exp_name
        self.md_path = self.folder_path+'/'+self.file_name+'.md'
        self.rec_path = self.folder_path+'/'+self.file_name+'_record.json'
        self.tdb = db = TinyDB(self.rec_path)
        self.md_head = '# Experiment Report\n## '+time.strftime('%Y年%m月%d日 %H:%M:%S\t')+ \
            self.exp_name+'\n- Log location: ```'+self.folder_path+'```\n'+ \
            '\n- Project location: ```'+self.curr_prj_path+'```'
        self.md_content_namelist = []
        self.md_content_dict = {} # [varname_key]={"type":"plot", "content":"/h/y"}) 
                # "type":"plot"(path, title, comment), "text"(content), "table"(varname_key)
        self.md_tail = ''

        self.plot_manager = {}


        self.md_write()
        atexit.register(self.exit_handler)

    def exit_handler(self):
        if 'time_cons' not in self.md_content_namelist:
            self.md_content_namelist.insert(0, 'time_cons')
        self.md_content_dict['time_cons']={"type":"text","content": \
            '## Total time consume:\t'+str(time.time()-self.init_time)}
        self.flush()
        self.md_write()
        print('Xreport已妥善记录')

    def md_write(self):
        content = ''
        # import pdb; pdb.set_trace()
        for name in self.md_content_namelist:
            content_d = self.md_content_dict[name]
            if content_d["type"] == "plot": # 一般只plot当前文件夹下的
                content += '\n\n---\n## Plot of '+content_d["title"]+'\n'+ \
                    '\n!['+content_d["title"]+']('+content_d["path"]+')'
            elif content_d["type"] == "text": # 一般只plot当前文件夹下的
                content += '\n'+content_d["content"]+'\n'
            elif content_d["type"] == "table": # 一般只plot当前文件夹下的
                content += '\n'+content_d["content"]+'\n'

        with open(self.md_path, 'w', encoding="utf-8") as mdf:
            mdf.write(self.md_head+'\n'+content+'\n'+self.md_tail)

    def discription(self, disc='这次实验主要目的是测试。'):
        self.md_head += '\n## 描述\n'+disc
        self.md_write()

    def flush(self, varname_key=''):
        if varname_key:
            self.tdb.table(varname_key).insert_multiple(self.record_buffer[varname_key])
            self.record_buffer[varname_key] = []
        else:
            for k, v  in self.record_buffer.items():
                self.tdb.table(k).insert_multiple(v)
            self.record_buffer.clear()

    def plot_update(self, varname_key, plot_fn=None, xlabel='', ylabel='', reverse_hist=True):
        if varname_key not in self.plot_manager.keys():
            self.plot_manager[varname_key] = {"index":1, 'gif_path_list':[]}
            self.md_content_namelist.append(varname_key)
            self.md_content_dict[varname_key]={"type":"plot","title":varname_key,"path":varname_key+('.gif' if reverse_hist else '.png')}
            # self.md_content_namelist.append({"type":"plot","title":varname_key,"path":varname_key+('.gif' if reverse_hist else '.png')})
        # [{'time': 1555496028.802764, 'data': 111}, {'time': 1555496034.8218799, 'data': 222}]
        values = self.tdb.table(varname_key).all()
        self.plot_manager[varname_key]['values']=[value['data'] for value in values]
        plot_dir = self.folder_path+'/'+varname_key+'_plots'
        if plot_fn is None:
            # import pdb; pdb.set_trace()
            plt.figure(varname_key)
            plt.gcf().clear()
            data = self.plot_manager[varname_key]['values']
            plt.plot(data, label=varname_key)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            if reverse_hist:
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(plot_dir+'/'+varname_key \
                        +'_'+str(self.plot_manager[varname_key]['index'])+'.png')
            plt.savefig(self.folder_path+'/'+varname_key+'.png')
            if len(data) >= 5:
                data = np.array(data)
                if varname_key+'_table' not in self.md_content_namelist:
                    self.md_content_namelist.insert(self.md_content_namelist.index(varname_key)+1, varname_key+'_table')
                self.md_content_dict[varname_key+'_table']={"type":"text",
                    "content": '### Statistics\n|Max |Min |Argmax |Argmin |1st |\n|---|---|---|---|---|\n|'+ \
                        str(data.max())+' |'+str(data.min())+' |'+str(data.argmax())+' |'+str(data.argmin())+' |'+str(data[0])+\
                        '|\n\n|Avg |Std |Avg[-5:] |Std[-5:] |\n|---|---|---|---|\n|'+ \
                        str(data.mean())+' |'+str(data.std())+' |'+str(data[-5:].mean())+' |'+str(data[-5:].std())+'|'
                    }

        else:
            # plot_tmp_img_path = plot_fn(self.plot_manager[varname_key]['values'][-1], inputtype='list')
            plot_tmp_img_path = plot_fn(self.plot_manager[varname_key]['values'][-1], inputtype='list', show=False)
            if reverse_hist:
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                mark_img = Image.open(plot_tmp_img_path)
                ImageDraw.Draw(mark_img).text((5, 5), str(self.plot_manager[varname_key]['index']), fill="#ff0000")
                mark_img.save(plot_dir+'/'+varname_key \
                        +'_'+str(self.plot_manager[varname_key]['index'])+'.png', 'png')
                # shutil.copyfile(plot_tmp_img_path, plot_dir+'/'+varname_key \
                #         +'_'+str(self.plot_manager[varname_key]['index'])+'.png')
                self.plot_manager[varname_key]['gif_path_list'].append(plot_dir+'/'+varname_key \
                        +'_'+str(self.plot_manager[varname_key]['index'])+'.png')
                # create_gif(self.plot_manager[varname_key]['gif_path_list'], self.folder_path+'/'+varname_key+'.gif')
            shutil.move(plot_tmp_img_path, self.folder_path+'/'+varname_key+'.png')
        if reverse_hist:
            create_gif(self.plot_manager[varname_key]['gif_path_list'], self.folder_path+'/'+varname_key+'.gif')
        self.plot_manager[varname_key]['index'] += 1
        self.md_write()

    '''
    plot_fn: None-不画图
    '''
    def record(self, varname_key, value, epoch=None, step=None, sync_every=20, plot_fn=None):
        if varname_key not in self.record_buffer.keys():
            self.record_buffer[varname_key] = []
        formatted_value = {}
        formatted_value['time']=time.time()
        formatted_value['data']=value
        if epoch is not None:
            formatted_value['epoch']=epoch
            if 'epoch_indicate' not in self.md_content_namelist:
                self.md_content_namelist.insert(0, 'epoch_indicate')
            self.md_content_dict['epoch_indicate']={"type":"text","content":'## Epochs Finished Counting\t'+str(epoch)}
        if step is not None:
            formatted_value['step']=step
        self.record_buffer[varname_key].append(formatted_value)

        if len(self.record_buffer[varname_key]) >= sync_every:
            self.flush(varname_key)

            # 在这里plot更新一次
            # self.plot_update(varname_key, plot_fn=plot_fn)
    
    # def record4all_write(self):
        # with open(self.rec_path, 'w', encoding="utf-8") as recf:
        #     json.dump(self.record_buffer, recf)

# xxx = Xreport()
# dic = xxx.__dict__
# print(dic)
#print(json.dumps(dic))