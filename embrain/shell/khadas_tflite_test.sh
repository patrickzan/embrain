#!/bin/bash

sshpass -p "khadas" scp ${1}/${2} root@192.168.1.2:/usr/test/tflite
sshpass -p "khadas" ssh root@192.168.1.2 "cd /usr/test/tflite;
    	  chmod ugo+x ./tools/linux_aarch64_benchmark_model_plus_flex;
    	  ./tools/linux_aarch64_benchmark_model_plus_flex --graph=${2} --num_threads=4 --enable_op_profiling=true > /usr/test/tflite/${2}.log;
    	  rm -rf *.tflite"

sshpass -p "khadas" scp root@192.168.1.2:/usr/test/tflite/${2}.log ${1}
sshpass -p "khadas" ssh root@192.168.1.2 "cd /usr/test/tflite; rm -rf *.log"
echo "Performance log saved to ${1}/${2}."