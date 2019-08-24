class XprintTorch:
    '''
    A tool for debuging tensorflow variables.
    '''

    def __init__(self):
        self.content = ''
        self.alreadyprint = False
        self.alreadyprintstructure = False
        self.alreadycollected = False
        self.watchmodel = None
        self.watchoptimizer = None

    def col(self, module=None, inputs=None):
        '''
        Use "xp.collect( )" to collect variables.
        '''
        if module is not None:
            self.content += '\n\n'+str(module)
            for k, v in module._parameters.items():
                self.content+='\n\t- has param **"'+k+'"** with shape '+str(v.shape)

        if inputs is not None:
            self.content += '\n- The input has shape of '+str(inputs.shape)
    
    def print(self, show=True, mustmanytimes=False):
        if mustmanytimes or not self.alreadyprint:
            self.alreadyprint = True
            if show:
                print(self.content)
            return self.content
        else:
            return ''

xpt = XprintTorch()

def p_stru(torch_module, treelevel=2):
    '''打印pytorch里面的模块层次结构
    treelevel是打印内部多少层,-1为全部层都打印
    可以用p_stru(model._module['xx'], 2)打印子层
    '''
    res_str = stru(torch_module, treelevel=treelevel)
    print(res_str)

def stru(torch_module, treelevel=-1, intend=1):
    r"""打印pytorch里面的模块层次结构

    treelevel是打印内部多少层,-1为全部层都打印
    
    可以用p_stru(model._modules['xx'], 2)打印子层
    """
    stru_str = ''
    stru_str += torch_module.__class__.__name__
    childs = list(torch_module._modules.keys())
    # stru_str += ':['+', '.join(childs)+']'
    stru_str += ':'+str(childs)
    for index, child_module in enumerate(torch_module.children()):
        if treelevel != 0:
            child_stru_str = stru(child_module, treelevel=treelevel-1, 
                intend=intend+1)
            stru_str += '\n'+'    '*intend+'%d.'%(index+1)+\
                child_stru_str+'    ('+childs[index]+')'
        else:
            child_stru_str = 'end'
    # print(stru_str)
    # if torch_module.__class__.__name__ == 'MaskRCNN':
    #     import pdb; pdb.set_trace()
    return stru_str

