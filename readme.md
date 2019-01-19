**Status:** Active (under active development)

Welcome to xtool! 
==================================
当前版本为0.1.2

## 安装方法
- 下载此仓库到本地
- ```cd xtool```
- ```pip install -e .```

## 使用方法
- ```from xtool.xp import xp```
- ```xp.feed(inputs)```
- ```xp.collect(locals()[, pre=''])```
- 支持返回真值
```xp.p('tensor_var'[, num=20, r=False])```
