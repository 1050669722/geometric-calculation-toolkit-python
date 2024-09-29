#!/bin/bash

# cd 到 release/
cd release

# conda init bash
# conda activate base

# 切换conda环境 到 base
source ~/Programs/miniconda3/bin/activate base

# 遍历递归删除已经存在的目录
for obj in `ls`
    do
        if [ -d $obj ]
            then
                rm -rf $obj
        fi
    done

# 复制模块 到 release
cp -r ../Welt ./

# 打包产生.whl文件
python setup.py bdist_wheel
