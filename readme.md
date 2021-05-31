**Status:** Active (under active development)

Welcome to xtool! 
==================================
版本: 0.1.3

## 特性
### show模块用于在code中方便地打印和输出各种格式的文字，表格，图像等直观结果
- [xtool.show.list)](https://www.github.com/xianpf/xtool) : 把


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
