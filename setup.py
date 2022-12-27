from setuptools import setup, find_packages

setup(name='embrain', version='0.1', packages=find_packages(),
      install_requires=[
        'PyYAML',
        'flatbuffers==2.0.7',
        'tensorflow_addons',
        'protobuf==3.19',
        'onnx',
        'onnxruntime-gpu',
        'torch',
        'tf2onnx',
        'onnxconverter-common',
        'scikit-learn==1.1.1',
        'scipy',
        'skl2onnx',
        'tflite',
        'tensorflow-gpu', 
        'rich',
        'onnx-simplifier', 
        'onnx-tf',
        'tensorflow-probability'
      ])
