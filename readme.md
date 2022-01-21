**Status:** Active (under active development)

Welcome to xtool! 
==================================
版本: 0.1.3

## 特性
### show模块用于在code中方便地打印和输出各种格式的文字，表格，图像等直观结果
- [xtool.show.list](https://www.github.com/xianpf/xtool)：把**list列表**罗列的内容输出为terminal表格，xlsx格式，markdown表格， panda表格等。
- [xtool.show.plot](https://www.github.com/xianpf/xtool)：使用plt输出
- [xtool.show.savepltfig](https://github.com/xianpf/xtool/blob/master/usefulCodes/save_plt_fig.py): 把histogram等figure的展示图存为numpy可即时编辑
- [xtool.ParamMeter](https://github.com/xianpf/xtool/blob/master/usefulCodes/paramMeters.py): 好用的参数收集器

### chrome模块用于使用chrome进行的一些自动化操作
- xtool.chrome.*


## 安装方法
- 直接在线安装
  - pip install git+https://www.github.com/xianpf/xtool
- 下载此仓库到本地
  - ```cd xtool```
  - ```pip install -e .```

## 使用方法
- ```from xtool.xp import xp```
- ```xp.feed(inputs)```
- ```xp.collect(locals()[, pre=''])```
- 支持返回真值
```xp.p('tensor_var'[, num=20, r=False])```
- 支持直接打印tensor内容
```xp.p(mytensor)```
- 若没有自动解析session，可以自己指定session： ```xp.set_sess(sess)```
