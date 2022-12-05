import onnx, os, logging, subprocess
import tensorflow as tf
from glob import glob
from onnxsim import simplify
from embrain.utils import onnx_to_tflite
logger = logging.getLogger(__name__)
pwd = os.path.dirname(os.path.abspath(__file__))

# onnx_model_path = '/home/pzan/Documents/vision/classification/alexnet/bvlcalexnet-12.onnx'
# onnx_model_path = '/home/pzan/Documents/vision/classification/vgg16/vgg16-12.onnx'

def embrain_run(model_zoo_dir, model_name):
    print(f'running {model_name}...')
    model_dir = os.path.join(model_zoo_dir, model_name)
    onnx_model_path = glob(os.path.join(model_dir, '*.onnx'))[0]
    # 1. load and simplify onnx
    onnx_model = onnx.load(onnx_model_path)
    simp_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"


    # 2. convert to and save tflite model
    tflite_model_dir = os.path.join(os.path.dirname(onnx_model_path), 'tflite_model')
    tflite_model_rep = onnx_to_tflite(simp_model, tflite_model_dir)
    tflite_model_path = os.path.join(os.path.dirname(onnx_model_path), os.path.basename(onnx_model_path).split('.')[0] + '.tflite')
    open(tflite_model_path, 'wb').write(tflite_model_rep)
    print(f'tflite model saved to {tflite_model_path}.')


    # 3. push to khadas board and test
    tflite_model_name = os.path.basename(tflite_model_path)
    tflite_model_dir = os.path.dirname(tflite_model_path)
    # embrain_dir = os.path.dirname(pwd)
    # shell_dir = os.path.join(embrain_dir, 'shell')
    # os.chdir(shell_dir)
    subprocess.call(['../shell/khadas_test.sh', tflite_model_dir, tflite_model_name])



# model_names = ['alexnet', 'vgg16', 'vgg19', 'googlenet', 'mobilenetv2', 'resnet50']
model_names = ['squeezenetv1', 'resnet50']
model_zoo_dir = '/home/pzan/Documents/vision/classification'

for model_name in model_names:
    embrain_run(model_zoo_dir, model_name)