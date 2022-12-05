#!/bin/bash

#tfmodel=bisenetv2-float_model_second_part_lite.tflite
#tfpath=/bsnn/users/gangli/proj/test/perfbench/bisenetv2_retrain-bb/tflite_result
# tfmodel=bvlcalexnet-12.tflite
#second_part_model.tflite
# tfpath=/home/pzan/Documents/vision/classification/alexnet


# adb shell "mkdir /userdata/pengzan/test/tflite"
# adb push /bsnn/users/gangli/tools/linux_aarch64_benchmark_model_plus_flex /userdata/gangli/test/tftmp/

# adb push ${tfpath}/${tfmodel} /userdata/pengzan/test/tflite/
sshpass -p "khadas" scp ${1}/${2} root@10.42.0.42:/usr/test/tflite
sshpass -p "khadas" ssh root@10.42.0.42 "cd /usr/test/tflite;
    	  chmod ugo+x ./tools/linux_aarch64_benchmark_model_plus_flex;
    	  ./tools/linux_aarch64_benchmark_model_plus_flex --graph=${2} --num_threads=4 --enable_op_profiling=true > /usr/test/tflite/${2}.log;
    	  rm -rf *.tflite"

sshpass -p "khadas" scp root@10.42.0.42:/usr/test/tflite/${2}.log ${1}
sshpass -p "khadas" ssh root@10.42.0.42 "cd /usr/test/tflite; rm -rf *.log"
echo "Performance log saved to ${1}/${2}."