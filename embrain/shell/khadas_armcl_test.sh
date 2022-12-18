#!/bin/bash

# 1: board ip
# 2: board path to shared libraries and executable
# 3: executable file name
# 4: scheduling order 
# 5: first partition point 
# 6: second partition point
# 7: local path to store results
sshpass -p khadas ssh root@${1} "cd ${2}; 
        export LD_LIBRARY_PATH=${2};
        ./${3} --threads=4 --threads2=2 --n=50 --total_cores=6 --partition_point=${5} --partition_point2=${6} --order=${4} > ${2}/${3}.log"

sshpass -p khadas scp root@${1}:${2}/${3}.log ${7}
