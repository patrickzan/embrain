import onnx
import tensorflow as tf 
from onnx_tf.backend import prepare
from onnx import version_converter

def onnx_to_tflite(onnx_model, tf_model_path):
    """
    This function converts onnx model to tflite model.
    """
    # onnx opset conversion
    opset = 13
    onnx_model = version_converter.convert_version(onnx_model, opset)

    # convert to tflite
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_rep = converter.convert()
    return tflite_rep


