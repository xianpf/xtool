from typing import List


'''
    prMt = ParamMeters(['loss', 'N0rate', 'tgt_num'])
    for ep in range(num_ep):
        prMt.init(['loss', 'N0rate', 'tgt_num'])
        for iter, batch in enumerate(data_loader):
            imgs, tgts = batch[0].to(device), {k:v.to(device) for k,v in batch[1].items()}
            predictions, loss = model(imgs, tgts)
            prMt.collect('loss', loss, len(imgs))
            prMt.collect('N0rate', pred_n0, len(pred_n0))
        prt(f"{datetime.datetime.now().strftime('%d-%H:%M:%S')} "
                f"Ep:{ep}/{num_ep}-{iter}/{len(data_loader)}\t| avg_loss:{prMt.avg('loss'):0.3f}, "
                f"tgt_num|")
'''

class ParamMeters():
    def __init__(self, vars:List[str] = []) -> None:
        self.data = dict()
        self.init(vars)
    
    def init(self, vars:List[str] = []):
        for var in vars:
            self.data[var] = []
            self.data[f'num_{var}'] = 0

    def collect(self, var:str, value, num=1):
        assert var in self.data, f"{var} is not recorded, please initialize it first."
        if isinstance(value, List):
            self.data[var].extend(value)
        else:
            self.data[var].append(value)
        self.data[f'num_{var}'] += num
        
    def avg(self, var:str):
        return sum(self.data[var]) / self.data[f'num_{var}']
