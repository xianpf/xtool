import tensorflow as tf
from matplotlib import pyplot as plt

class Xprint:
    '''
    A tool for debuging tensorflow variables.
    '''

    def __init__(self, sess=None):
        self.sess = sess
        self.collected_var = dict()
        self.feed_dict = None

    def collect(self, var_dict, pre='', auto_sess=True):
        '''
        Use "xp.collect(locals())" to collect variables.
        '''
        # self.collected_var.update(var_dict)
        for k,v in var_dict.items():
            if auto_sess and isinstance(v, tf.Session):
                self.sess = v
            # if isinstance(v, tf.Tensor):
            self.collected_var.update({pre+k:v})
                
    def feed(self, inputs):
        '''
        input the feed values here.
        '''
        self.feed_dict = inputs

    def p(self, variables, num=10, r=False):
        '''
        print
        '''
        if variables is not list:
            variables = [variables]
        assert self.sess is not None, \
            "please set the sess first. Use set_sess()"
        for var in variables:
            var_name = 'unknown'
            if isinstance(var, str):
                var_name = var
                assert var_name in self.collected_var.keys(), \
                    "This is not collected or not a tensor."
                var = self.collected_var[var_name]
            assert isinstance(var, tf.Tensor), "This is not a tensor."
            var_value = self.sess.run(var, feed_dict=self.feed_dict)
            if r:
                return var_value
            else:
                result = self.print(var_value, num)
                print(var_name+' has a shape of '+str(var_value.shape)+ \
                ', its value is:', result)

    def set_sess(self, sess):
        self.sess = sess

    def print(self,val, num):
        # print(name+':\n', val, type(val), len(val.shape))
        print_str = ''
        if len(val.shape) and val.shape[0] > num + 1:
            print_str += '\n[' + ' '.join([self.print(v,num) for v in val[:num]]) + '\n...\n'+ \
                '  '.join([self.print(v,num) for v in val[-num:]]) + ']'
        else:
            print_str = str(val)
        return print_str

    def g(self, var_name):
        assert var_name in self.collected_var.keys(), \
            "This is not collected."
        return self.collected_var[var_name]

class XprintTorch:
    '''
    A tool for debuging tensorflow variables.
    '''

    def __init__(self):
        self.content = ''

    def col(self, module=None, inputs=None):
        '''
        Use "xp.collect( )" to collect variables.
        '''
        if module is not None:
            self.content += '\n'+str(module)
            for k, v in module._parameters.items():
                self.content+='\n\thas param '+k+' with shape '+str(v.shape)

        if inputs is not None:
            self.content += '\n The input has shape of '+str(inputs.shape)
    
    def print(self, show=True):
        if show:
            print(self.content)
        return self.content
        
class Xwatch:
    '''
    A tool for debuging tensorflow variables.
    '''

    def __init__(self):
        self.variables = dict()

    def collect(self, name, var):
        self.variables[name] = var

    def get(self, name):
        return self.variables[name]

class Xshow:
    '''
    A tool for debuging tensorflow variables.
    '''

    def __init__(self):
        self.variables = dict()

    def collect(self, name, var):
        self.variables[name] = var

    def get(self, name):
        return self.variables[name]

class Xstat():
    def __init__(self, dtype=None):
        pass
    
    def collect_data(self, one_data):
        pass



xp = Xprint()
xpt = XprintTorch()
xw = Xwatch()
# xs = Xshow()

def xshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def test():
    xst = Xstat()
    print(test)

test()