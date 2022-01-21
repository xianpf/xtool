from typing import List


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
