"""Tests for converter"""
import collections

import pytest
import timm
from torch import nn

from model_converter import Converter


def create_timm_classifier(encoder_name, num_classes, in_chans) -> nn.Module:
    return timm.create_model(
        encoder_name, pretrained=True, num_classes=num_classes, in_chans=in_chans
    )


class ClsReg2Model(nn.Module):
    """A neural network that combines classification and regression heads.

    Parameters
    ----------
    nn : [nn.Module]
        Core feature extractor model that takes as input images and outputs feature
        vector, e.g. of dimension Bx2048x7x7
    """

    Output = collections.namedtuple("output", ["cls", "reg_xy", "reg_wh"])

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.head_xy = nn.Linear(in_features=self.core.num_features, out_features=2)
        self.head_wh = nn.Linear(in_features=self.core.num_features, out_features=2)

    def forward(self, x):
        """Forward pass for train/inference."""
        features = self.core.forward_features(x)
        cls = self.core.forward_head(features)

        reg_input = self.core.forward_head(features, pre_logits=True)
        print(self.core.num_features)
        print(reg_input.shape)
        reg_xy = self.head_xy(reg_input)
        reg_wh = self.head_wh(reg_input)
        return self.Output(cls=cls, reg_xy=reg_xy, reg_wh=reg_wh)


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

    for model_name in expected_out_models:
        model_path = tmp_path / model_name
        assert model_path.is_file(), f"{model_name=} does not exist"


@pytest.mark.parametrize(
    ("encoder_name", "num_classes", "in_size", "in_chans"),
    [
        ("mobilenetv3_large_100", 10, [224, 224], 3),
        ("resnet50", 10, [224, 224], 3),
    ],
)
def test_classifier_w_regression(
    tmp_path, encoder_name, num_classes, in_size, in_chans
):
    expected_out_models = {"model.onnx", "model.mlmodel", "model.tflite"}

    # initialize converter
    converter = Converter(save_dir=tmp_path, simplify_exported_model=True)

    # create encoder
    classifier = create_timm_classifier(encoder_name, num_classes, in_chans)

    # create source model
    model_to_convert = ClsReg2Model(classifier)

    # convert
    converter.convert(
        model_to_convert,
        batch_size=1,
        input_size=in_size,
        channels=in_chans,
        force=True,
        fmt="tflite_coreml",
    )

    for model_name in expected_out_models:
        model_path = tmp_path / model_name
        assert model_path.is_file(), f"{model_name=} does not exist"
