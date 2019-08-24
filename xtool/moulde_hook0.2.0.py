# version 0.2.0

import torch, pickle
import numpy as np
from collections import OrderedDict
import inspect
# from xtool.xshow import PlotShow, channel2grayRGB
# RCNNFeatureshow = PlotShow('RCNNFeatures', frame_size=(1024,1024))


# # xpf_names = [n for n,p in self.named_children()]
# xpf_names = [n for n,p in self.named_modules(prefix='xxxx')]
# RCNNFeatureshow.update(channel2grayRGB(input[0].cpu().data.numpy()[0][0]))

# def xpf_name_this_module(self, prefix=''):
#     self.xpf_moduld_name = 
#     for name, module in self._modules.items():
#         if module is not None:
#             module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
feature_buf = dict()
forward_codes = dict()
steps = []
record_switcher = False
save_path = '/home/xianr/TurboProjects/maskRL/save_features_train.pkl'
def turn_on_record(path=None):
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
    general_buf = {
        'feature_buf':feature_buf,
        'forward_codes': forward_codes, 
        'steps':steps}
    with open(save_path, 'wb') as f:
        pickle.dump(general_buf, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(save_path, 'rb') as f:
        # feature_buf_read = pickle.load(f)
    
def my_module_hook(module, input_f, output_f):
    if not record_switcher:
        return
    else:
        # if not module.xpf_moduld_name:
        if 'xpf_moduld_name' not in module.__dict__.keys():
            print('Error! module.xpf_moduld_name is not defined')
            import pdb; pdb.set_trace()
        else:
            # print('my_module_hook', module.xpf_moduld_name)
            code = str(module)+'@\n'+inspect.getsource(module.forward)
            # save_features(module.xpf_moduld_name, input_f[0], code=code)
            save_features(module.xpf_moduld_name, input_f, code=code)

def save_features(name, input_f, code=None, ret=False):
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
        self.feature_buf_key = ''
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

class FeatureRoot(FeatureNode):
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

# rootFeature = FeatureRoot(save_path)
# rootFeature.parse_feature()
# res = [print(name == node.name, name,'\t\t', node.name) for name, node in rootFeature.child_nodes()]

