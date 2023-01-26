from typing import Optional

import logging
import pathlib

import onnx
import torch
from torch import nn

from model_converter.model_utils import (
    check_onnx_model,
    export_model_onnx,
    load_model_weights,
)

try:
    import keras  # pylint: disable=W0611
    import tensorflow as tf

    # Disable GPU memory allocation
    tf.config.set_visible_devices([], "GPU")
except ImportError:
    tf = None

try:
    import coremltools
except ImportError:
    coremltools = None

try:
    import onnx2keras
except ImportError:
    onnx2keras = None

LOG = logging.getLogger(__name__)


class Converter:
    """Conversion wrapper class"""

    def __init__(self, save_dir, simplify_exported_model=False):
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        self.save_dir = save_dir

        self.onnx_path = self.save_dir / "model.onnx"
        self.keras_path = self.save_dir / "model.h5"
        self.tflite_path = self.save_dir / "model.tflite"
        self.coreml_path = self.save_dir / "model.mlmodel"
        self.simplify_exported_model = simplify_exported_model

    def to_onnx(self, torch_model: nn.Module, batch_size, input_size, channels):
        """Torch -> ONNX"""
        w, h = input_size
        export_model_onnx(
            torch_model,
            str(self.onnx_path),
            batch_size,
            h,
            w,
            channels,
            self.simplify_exported_model,
        )
        LOG.info("ONNX model is saved to %s", self.onnx_path)
        check_onnx_model(str(self.onnx_path))
        return [self.onnx_path]

    def to_keras(self, torch_model: nn.Module, batch_size, input_size, channels):
        """Torch -> ONNX -> Keras"""
        if tf is None:
            raise RuntimeError("Tensorflow is required for conversion")
        if onnx2keras is None:
            raise RuntimeError("Onnx2keras is required for conversion")

        if not self.onnx_path.exists():
            self.to_onnx(torch_model, batch_size, input_size, channels)

        # Load ONNX model
        onnx_model = onnx.load(str(self.onnx_path))

        # Call the converter (input - is the main model input name,
        # can be different for your model)
        keras_model = onnx2keras.onnx_to_keras(
            onnx_model, ["input"], change_ordering=True
        )
        keras_model.save(str(self.keras_path))
        LOG.info("Keras model is saved to %s", self.keras_path)
        return [self.onnx_path, self.keras_path]

    def to_tflite(self, torch_model: nn.Module, batch_size, input_size, channels):
        """Torch -> ONNX -> Keras -> TFLite"""
        if tf is None:
            raise RuntimeError("Tensorflow is required for conversion")

        if not self.keras_path.exists():
            self.to_keras(torch_model, batch_size, input_size, channels)

        keras_model = tf.keras.models.load_model(self.keras_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # Possible optimizations
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        self.tflite_path.write_bytes(tflite_model)
        LOG.info("TFlite model is saved to %s", self.tflite_path)
        return [self.onnx_path, self.tflite_path]

    def to_coreml(self, torch_model: nn.Module, batch_size, input_size, channels):
        """Torch -> TorchScript -> CoreML"""
        if coremltools is None:
            raise RuntimeError("Coremltools is required for conversion")

        example_input = torch.rand(batch_size, channels, *input_size)
        traced_torch_model = torch.jit.trace(torch_model, example_input)
        example_torch_output = traced_torch_model(example_input)

        coreml_model = coremltools.convert(
            traced_torch_model,
            source="pytorch",
            inputs=[coremltools.TensorType(shape=[batch_size, channels, *input_size])],
        )
        coreml_spec = coreml_model.get_spec()

        if hasattr(torch_model, "Output"):
            # Rename output nodes
            output_names = torch_model.Output._fields
            if len(output_names) > 1:
                for i, output_name in enumerate(output_names):
                    output_shape = list(example_torch_output[i].shape)
                    coreml_spec.description.output[i].type.multiArrayType.shape.extend(
                        output_shape
                    )
                    coremltools.utils.rename_feature(
                        coreml_spec, coreml_spec.description.output[i].name, output_name
                    )
            else:
                output_shape = list(example_torch_output[0].shape)
                coreml_spec.description.output[0].type.multiArrayType.shape.extend(
                    output_shape
                )
                coremltools.utils.rename_feature(
                    coreml_spec, coreml_spec.description.output[0].name, output_names[0]
                )

        coreml_model = coremltools.models.MLModel(coreml_spec)
        coreml_model.save(self.coreml_path)
        LOG.info("CoreML model is saved to %s", self.coreml_path)

        return [self.coreml_path]

    def to_tflite_coreml(
        self, torch_model: nn.Module, batch_size, input_size, channels
    ):
        tflite_output_models = self.to_tflite(
            torch_model, batch_size, input_size, channels
        )
        coreml_output_models = self.to_coreml(
            torch_model, batch_size, input_size, channels
        )

        # remove duplicates
        coreml_output_models = set(coreml_output_models) - set(tflite_output_models)

        return tflite_output_models + list(coreml_output_models)

    def convert(
        self,
        torch_model: nn.Module,
        batch_size: int,
        input_size: list,
        channels: int,
        fmt: str,
        force: bool = True,
        torch_weights: Optional[pathlib.Path] = None,
    ):
        if torch_weights is not None:
            load_model_weights(torch_model, torch_weights, fuse_bn=True)
        if force:
            if self.coreml_path.exists():
                self.coreml_path.unlink()
            if self.tflite_path.exists():
                self.tflite_path.unlink()
            if self.keras_path.exists():
                self.keras_path.unlink()
            if self.onnx_path.exists():
                self.onnx_path.unlink()

        func = getattr(self, f"to_{fmt}")
        return func(torch_model, batch_size, input_size, channels)
