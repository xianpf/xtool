# version 1.0.0

import torch, pickle, datetime, os
import numpy as np
from collections import OrderedDict
import inspect
from pymongo import MongoClient
from bson.binary import Binary

# feature_buf = dict()
# forward_codes = dict()
paths_order_recorder = []
appear_index = 0
record_switcher = False
save_path = '/home/xianr/data/trainlogs/feature_tracker'
collection = None

class RecordNode():
    def __init__(self, name, nnFuncname, code, input_4d_tensor, output_4d_tensor, 
                weights, save_dir, appear_index, good_tensor=True):
        self.name = name
        self.nnFuncname = nnFuncname
        self.code = code
        self.input_4d_tensor = input_4d_tensor
        self.output_4d_tensor = output_4d_tensor
        self.weights = weights
        self.save_dir = save_dir
        self.appear_index = appear_index
        self.good_tensor = good_tensor
        # print('good')
    
    def dump(self, path_order_recorder):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        filename = self.save_dir + '/' + self.name + '_.pkl'
        path_order_recorder.append(filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)



def turn_on_record(path=None):
    global record_switcher
    global save_path
    if path:
        save_path = path
    record_switcher = True
    
def turn_on_record_mongo_test(save_name=None):
    global record_switcher
    global collection

    record_switcher = True
    client = MongoClient('localhost', 27017)
    db = client['feature_storage']
    if not save_name:
        save_name = 'default_feature'
    collection = db[save_name]
    
def turn_on_record_old(path=None):
    global record_switcher
    global save_path
    if path:
        save_path = path
    record_switcher = True
    
def turn_off_record():
    global record_switcher
    if not record_switcher:
        print('Warning! record_switcher is off already!')
        import pdb; pdb.set_trace()
    record_switcher = False
    # general_buf = {
    #     'feature_buf':feature_buf,
    #     'forward_codes': forward_codes, 
    #     'steps':steps}
    with open(save_path+'/paths_order_recorder.pkl', 'wb') as f:
        pickle.dump(paths_order_recorder, f, protocol=pickle.HIGHEST_PROTOCOL)
    # # with open(save_path, 'rb') as f:
    #     # feature_buf_read = pickle.load(f)
    
def my_forward_pre_hook(module, input_f):
    paths_order_recorder.append('__pre__'+(module.xpf_moduld_name if 'xpf_moduld_name'in module.__dict__.keys() else 'NoxpfName')+'\t@\t'+str(input_f[0].shape if isinstance(input_f[0], torch.Tensor) else type(input_f[0]))+'\t@\t'+str(module))

def my_module_hook(module, input_f, output_f):
    if not record_switcher:
        return
    else:
        if 'xpf_moduld_name' not in module.__dict__.keys():
            print('Error! module.xpf_moduld_name is not defined')
            import pdb; pdb.set_trace()
        else:
            global appear_index
            name = module.xpf_moduld_name
            nnFuncname = str(module)
            code = inspect.getsource(module.forward)
            weights = module._parameters
            print('name: '+name+'\t', input_f[0].shape if isinstance(input_f[0], torch.Tensor) else type(input_f[0]))
            # import pdb; pdb.set_trace()
            save_features(name, nnFuncname, code, input_f, output_f, 
                weights, save_path, appear_index)
            appear_index += 1

def save_features(name, nnFuncname, code, input_f, output_f, 
                weights, save_dir, appear_index):
    if isinstance(input_f, torch.Tensor):
        # print('pure tensor shape', input_f.shape)
        # feature_buf[name] = input_f.cpu().data.numpy()
        tmprecnode = RecordNode(name, nnFuncname, code, input_f.cpu().data.numpy(), output_f, 
                weights, save_dir, appear_index, good_tensor=True)
        tmprecnode.dump(paths_order_recorder)
    else:
        if (isinstance(input_f, list) or isinstance(input_f, tuple)):
            # print('list data with len of',len(input_f), len(output_f))
            for i, input_ele in enumerate(input_f):
                newname = name+'-->'+type(input_f).__name__+'_'+str(i)+'.' if len(input_f) > 1 else name
                newcode = '['+type(input_f).__name__+':'+str(i)+'of'+str(len(input_f))+\
                    ']Parent code | '+code if len(input_f) > 1 else code
                # if len(input_f) == len(output_f):
                #     newoutput_f = [out.cpu().data.numpy() for out in output_f]
                # newoutput_f = [out.cpu().data.numpy() for out in output_f] if isinstance(output_f[0], torch.Tensor) else output_f
                save_features(newname, nnFuncname, newcode, input_f[i], output_f, 
                    weights, save_path, appear_index)
        elif input_f.__class__.__name__ == 'ImageList':
            newname = name+'-->'+type(input_f).__name__+'.'
            newcode = '['+type(input_f).__name__+']with "image_sizes":= '+str(input_f.image_sizes)+' |'+code
            save_features(newname, nnFuncname, newcode, input_f.tensors, output_f, 
                    weights, save_path, appear_index)
        elif input_f.__class__.__name__ == 'BoxList':
            print('Caution! A [BoxList] block unhandled!')
            name = name[:-1] + '[BoxList].'
            # pass
        elif input_f.__class__.__name__ == 'NoneType':
            print('Caution! A [BoxList] block unhandled!')
            name = name[:-1] + '[BoxList].'
            # pass
        #     else:
        #         print('Error! 未指定的类型')
        #         import pdb; pdb.set_trace()

        else:
            print('Error! 未指定的类型')
            import pdb; pdb.set_trace()
        
        if isinstance(input_f, tuple) and len(input_f)==1:
            pass
        else:
            tmprecnode = RecordNode(name, nnFuncname, code, input_f, output_f, 
                    weights, save_dir, appear_index, good_tensor=False)
            tmprecnode.dump(paths_order_recorder)

def save_features_strange_repeat(name, nnFuncname, code, input_f, output_f, 
                weights, save_dir, appear_index):
    # print('name type:'+name+'\t', type(input_f))
    if isinstance(input_f, torch.Tensor):
        # print('pure tensor shape', input_f.shape)
        # feature_buf[name] = input_f.cpu().data.numpy()
        tmprecnode = RecordNode(name, nnFuncname, code, input_f.cpu().data.numpy(), output_f, 
                weights, save_dir, appear_index, good_tensor=True)
        tmprecnode.dump(paths_order_recorder)
    else:
        if (isinstance(input_f, list) or isinstance(input_f, tuple)):
            print('list data with len of',len(input_f), len(output_f))
            for i, input_ele in enumerate(input_f):
                newname = name+'-->'+type(input_f).__name__+'_'+str(i)+'.' if len(input_f) > 1 else name
                newcode = '['+type(input_f).__name__+':'+str(i)+'of'+str(len(input_f))+\
                    ']Parent code | '+code if len(input_f) > 1 else code
                # if len(input_f) == len(output_f):
                #     newoutput_f = [out.cpu().data.numpy() for out in output_f]
                # newoutput_f = [out.cpu().data.numpy() for out in output_f] if isinstance(output_f[0], torch.Tensor) else output_f
                save_features(newname, nnFuncname, newcode, input_f[i], output_f, 
                    weights, save_path, appear_index)
        elif input_f.__class__.__name__ == 'ImageList':
            newname = name+'-->'+type(input_f).__name__+'.'
            newcode = '['+type(input_f).__name__+']with "image_sizes":= '+str(input_f.image_sizes)+' |'+code
            save_features(newname, nnFuncname, newcode, input_f.tensors, output_f, 
                    weights, save_path, appear_index)
        elif input_f.__class__.__name__ == 'BoxList':
            print('Caution! A [BoxList] block unhandled!')
            name = name[:-1] + '[BoxList].'
            # pass
        elif input_f.__class__.__name__ == 'NoneType':
            print('Caution! A [BoxList] block unhandled!')
            name = name[:-1] + '[BoxList].'
            # pass
        #     else:
        #         print('Error! 未指定的类型')
        #         import pdb; pdb.set_trace()

        else:
            print('Error! 未指定的类型')
            import pdb; pdb.set_trace()
        
        # if isinstance(input_f, tuple) and len(input_f)==1:
        #     pass
        # else:
        #     tmprecnode = RecordNode(name, nnFuncname, code, input_f, output_f, 
        #             weights, save_dir, appear_index, good_tensor=False)
        #     tmprecnode.dump(paths_order_recorder)

def save_features_deal_different(name, input_f, code=None, ret=False):
    steps.append(name)
    print('llllllllll:'+name+'\t', type(input_f))
    if isinstance(input_f, torch.Tensor):
        print('pure tensor shape', input_f.shape)
        feature_buf[name] = input_f.cpu().data.numpy()
        if code is not None:
            forward_codes[name] = code
        if ret:
            return feature_buf[name]
    else:
        if (isinstance(input_f, list) or isinstance(input_f, tuple)):
            print('list data with len of',len(input_f))
            for i, input_ele in enumerate(input_f):
                newname = name+'-->'+type(input_f).__name__+'_'+str(i)+'.' if len(input_f) > 1 else name
                newcode = '['+type(input_f).__name__+':'+str(i)+'of'+str(len(input_f))+\
                    ']Parent code | '+code if len(input_f) > 1 else code
                child_result = save_features(newname, input_ele, code=newcode, ret=True)
                if (child_result is not None) and (name not in feature_buf.keys()):
                    feature_buf[name] = child_result
        elif input_f.__class__.__name__ == 'ImageList':
            # import pdb; pdb.set_trace()
            newname = name+'-->'+type(input_f).__name__+'.'
            newcode = '['+type(input_f).__name__+']with "image_sizes":= '+str(input_f.image_sizes)+' |'+code
            child_result = save_features(newname, input_f.tensors, code=newcode, ret=True)
            if name not in feature_buf.keys():
                feature_buf[name] = child_result
        elif input_f.__class__.__name__ == 'BoxList':
            # import pdb; pdb.set_trace()
            child_result = None
            return
        elif input_f.__class__.__name__ == 'NoneType':
            # import pdb; pdb.set_trace()
            child_result = None
            return
        else:
            print('Error! 未指定的类型')
            import pdb; pdb.set_trace()
        if ret:
            if child_result is None:
                # if name not in feature_buf.keys():
                #     import pdb; pdb.set_trace()
                return
            else:
                if code is not None:
                    forward_codes[name] = code
                return feature_buf[name]

    # else:
    #     print('Error! 未指定的类型')
    #     import pdb; pdb.set_trace()
    
    if code is not None:
        forward_codes[name] = code
    # if ret:
    #     # if child_result is not None:
    #     if name not in feature_buf.keys():
    #         # print(child_result is None)
    #         import pdb; pdb.set_trace()
    #     return feature_buf[name]

def save_features_for_pytorch_maskrcnn(name, input_f_0, code=None):
    steps.append(name)
    if code is not None:
        forward_codes[name] = code
    if isinstance(input_f_0, torch.Tensor):
        feature_buf[name] = input_f_0.cpu().data.numpy()
    elif isinstance(input_f_0, list) and isinstance(input_f_0[0], torch.Tensor):
        feature_buf[name] = [input_f_0_ele.cpu().data.numpy() for input_f_0_ele in input_f_0]
    else:
        print('Error! 未指定的类型')
        import pdb; pdb.set_trace()
    

class FeatureNode():
    def __init__(self, name, parent_node, root_node, sibling_index=0):
        self.detail_path = ''
        self.name = name
        self.parent_node = parent_node
        self.root_node = root_node
        self.children_nodes = OrderedDict()
        self.child_index = 0
        self.sibling_index = sibling_index
        self.channel_index = 0
        self.channel_num = 0

    def child_nodes(self, prefix=''):
        if len(self.children_nodes) > 0:
            for name, node in self.children_nodes.items():
                yield prefix + ('.' if prefix else '') + name, node
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for c_name, c_node in node.child_nodes(prefix=submodule_prefix):
                    yield c_name, c_node
    
    def load_details(self):
        with open(self.detail_path, 'rb') as f:
            detail = pickle.load(f)
        return detail

class FeatureRootOld(FeatureNode):
    '''Usage:
    rootFeature = FeatureRoot(save_path)
    rootFeature.parse_feature()
    res = [print(name == node.name, name,'\t\t', node.name) for name, node in rootFeature.child_nodes()]'''
    def __init__(self, feature_buf_path):
        super(FeatureRoot, self).__init__('root', self, self)
        with open(feature_buf_path, 'rb') as f:
            self.general_buf = pickle.load(f)
            self.feature_buf = self.general_buf['feature_buf']
            self.forward_codes = self.general_buf['forward_codes']

    def __getattr__(self, name):
        import pdb; pdb.set_trace()
        super.__getattr__(name)

    def parse_feature(self):
        for key in self.feature_buf.keys():
            # print('key', key)
            # if '-->ImageList' in key:
            #     import pdb; pdb.set_trace()
            route = key.split('.')
            route = route if route[-1] else route[:-1]

            itern = self
            for i, part in enumerate(route):
                if part not in itern.children_nodes.keys():
                    itern.children_nodes[part] = FeatureNode('.'.join(route[:i+1]),\
                         itern, self, sibling_index=len(itern.children_nodes))
                itern = itern.children_nodes[part]
            itern.feature_buf_key = key

class FeatureRoot(FeatureNode):
    def __init__(self, feature_buf_path):
        super(FeatureRoot, self).__init__('root', self, self)
        with open(save_path+'/paths_order_recorder.pkl', 'rb') as f:
            self.paths_order_recorder = pickle.load(f)
        # self.keys = [p.split('/')[-1].split('_.pkl')[0] for p in self.paths_order_recorder]
        kkey = []
        for p in self.paths_order_recorder:
            if '@' not in p:
                kkey.append(p.split('/')[-1].split('_.pkl')[0])
        self.keys = kkey
        # import pdb; pdb.set_trace()

        print('record number:', len(self.keys))
        for key in self.paths_order_recorder:
            # print('Name:', key)
            keyname = key.split('@')[0].split('/')[-1]+key.split('@')[1] if '@' in key else key.split('/')[-1].split('_.pkl')[0]
            print(keyname)

    def __getattr__(self, name):
        import pdb; pdb.set_trace()
        super.__getattr__(name)

    def parse_feature(self):
        for i, key in enumerate(self.keys):
            print('xxxxxxxxxxxxxxxxx', key)
            route = key.split('.')
            route = route if route[-1] else route[:-1]

            itern = self
            for i, part in enumerate(route):
                if part not in itern.children_nodes.keys():
                    itern.children_nodes[part] = FeatureNode('.'.join(route[:i+1]),\
                         itern, self, sibling_index=len(itern.children_nodes))
                    # print('Node:', '.'.join(route[:i+1]))
                itern = itern.children_nodes[part]
            # itern.detail_path = self.paths_order_recorder[i]
            # global save_path
            itern.detail_path = save_path + '/' + key + '_.pkl'


