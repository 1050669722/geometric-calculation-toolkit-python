#!/bin/bash

# cd 到 release/
cd release

# 切换conda环境 到 undergroundParkBote
source ~/Programs/miniconda3/bin/activate undergroundParkBote

# 删除已经存在的Welt
pip uninstall Welt -y

# cd 到 dist/
cd dist

# 安装.whl文件
for obj in `ls`
    do
        if [ -e *.whl ]
            then
                pip install $obj
        fi
    done
