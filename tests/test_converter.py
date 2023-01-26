"""Tests for converter"""
import pytest
import timm
from torch import nn

from model_converter import Converter


def create_timm_classifier(encoder_name, num_classes, in_chans) -> nn.Module:
    return timm.create_model(
        encoder_name, pretrained=True, num_classes=num_classes, in_chans=in_chans
    )


@pytest.mark.parametrize(
    ("encoder_name", "num_classes", "in_size", "in_chans"),
    [
        ("mobilenetv3_large_100", 10, [224, 224], 3),
        ("resnet50", 10, [224, 224], 3),
    ],
)
def test_classifier(tmp_path, encoder_name, num_classes, in_size, in_chans):
    expected_out_models = {"model.onnx", "model.mlmodel", "model.tflite"}

    # initialize converter
    converter = Converter(save_dir=tmp_path, simplify_exported_model=True)

    # create source model
    model_to_convert = create_timm_classifier(encoder_name, num_classes, in_chans)

    # convert
    converter.convert(
        model_to_convert,
        batch_size=1,
        input_size=in_size,
        channels=in_chans,
        force=True,
        fmt="tflite_coreml",
    )

    p = tmp_path.glob("*")
    out_files = {x.name for x in p if x.is_file()}
    assert expected_out_models.issubset(out_files)
