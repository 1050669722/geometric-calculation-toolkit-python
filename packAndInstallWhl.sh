#!/bin/bash

curDir=$(pwd)

cd $curDir
bash pack.sh

cd $curDir
bash installWhl.sh
