import onnx, os, logging, subprocess
import tensorflow as tf
from glob import glob
from onnxsim import simplify
from embrain.utils import onnx_to_tflite
logger = logging.getLogger(__name__)
pwd = os.path.dirname(os.path.abspath(__file__))

def embrain_run(model_zoo_dir, model_name, model_type='classification'):
    print(f'running {model_name}...')
    model_dir = os.path.join(model_zoo_dir, model_name) if model_type == 'classification' else os.path.join(model_zoo_dir, model_name, 'model')
    onnx_model_path = sorted(glob(os.path.join(model_dir, '*.onnx')), key=lambda x: - int(os.path.splitext(x)[0].split('-')[-1]))[0]
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
    subprocess.call(['../shell/khadas_tflite_test.sh', tflite_model_dir, tflite_model_name])

# refer to link to download models in onnx. https://github.com/onnx/models
# model_names = []

# classification model zoo
cls_model_names = ['alexnet', 'vgg16', 'vgg19', 'googlenet', 'mobilenetv2', 
               'squeezenetv1', 'inceptionv1', 'inceptionv2', 'resnet50', 
               'shufflenetv1', 'shufflenetv2', 'densenet', 'rcnn', 
               'efficientnet', 'zfnet']
cls_model_zoo_dir = '/home/pzan/deepnn/vision/classification'
cls_model_type = 'classification'

# detection model zoo
det_model_zoo_dir = '/home/pzan/deepnn/vision/detection'
det_model_names = os.listdir(det_model_zoo_dir)
det_model_type = 'detection'

failed_cases = []
for model_name in det_model_names:
    try:
        embrain_run(det_model_zoo_dir, model_name)
    except:
        failed_cases.append(model_name)
print(f"Failed cases: {failed_cases}.")

