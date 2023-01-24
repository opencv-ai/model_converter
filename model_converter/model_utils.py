import copy
import logging
import pathlib
from collections import OrderedDict

import onnx
import torch
from torch import nn

LOG = logging.getLogger(__name__)


def simplify_onnx_model(
    onnx_model_path: pathlib.Path, simplified_model_checks: int = 5
):
    """Simplify the structure of ONNX graph using onnxsim"""
    from onnxsim import simplify

    LOG.info("Trying to simplify converted model")
    LOG.info("Loading original model")
    onnx_model = onnx.load(str(onnx_model_path))

    LOG.info("Simplifying ONNX model...")
    LOG.info(
        "Simplified model output will be checked %d times against original "
        "model output",
        simplified_model_checks,
    )
    simplified_model, are_checks_passed = simplify(onnx_model, simplified_model_checks)

    if are_checks_passed:
        LOG.info("Checks are passed")
    else:
        raise RuntimeError("Simplified model did not pass checks")
    onnx.save(simplified_model, str(onnx_model_path))


def export_model_onnx(
    model, save_path, batch_size, height, width, channels, simplify_exported_model
):
    """Convert Torch model to ONNX"""
    dummy_input = torch.randn(batch_size, channels, height, width)
    input_names = ["input"]
    output = model(dummy_input)
    if isinstance(output, dict):
        output_names = list(output.keys())
    elif isinstance(output, (list, tuple)):
        output_names = [f"output_{x}" for x in range(len(output))]
    else:
        output_names = ["output"]
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        verbose=False,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
    )
    if simplify_exported_model:
        simplify_onnx_model(save_path)


def check_onnx_model(onnx_model_path):
    """Check correctness of ONNX model"""
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)


def fuse_bn_sequential(block):
    """
    Take a sequential block and fuse the batch normalization with convolution
    :param block: nn.Sequential
    :return: nn.Sequential. Converted block
    """
    if not isinstance(block, nn.Sequential) or len(list(block.children())) <= 1:
        return block
    stack = []
    for name, m in block.named_children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1][1], (nn.Conv2d, nn.ConvTranspose2d)):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1][1].state_dict()
                out_axis = 0
                if isinstance(stack[-1][1], nn.ConvTranspose2d):
                    out_axis = 1

                # BatchNorm params
                eps = m.eps
                mu = bn_st_dict["running_mean"]
                var = bn_st_dict["running_var"]
                gamma = bn_st_dict["weight"]

                if "bias" in bn_st_dict:
                    beta = bn_st_dict["bias"]
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict["weight"]
                if "bias" in conv_st_dict:
                    bias = conv_st_dict["bias"]
                else:
                    bias = torch.zeros(W.size(out_axis)).float().to(gamma.device)

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(out_axis, -1)).transpose(out_axis, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1][1].weight.data.copy_(W)
                if stack[-1][1].bias is None:
                    stack[-1][1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1][1].bias.data.copy_(bias)
            else:
                stack.append((name, m))
        else:
            stack.append((name, m))

    return nn.Sequential(OrderedDict(stack))


def fuse_bn_recursively(model):
    """Fuse batch normalization with convolution"""
    for module_name in model._modules:  # pylint: disable=W0212
        model._modules[module_name] = fuse_bn_sequential(  # pylint: disable=W0212
            model._modules[module_name]
        )  # pylint: disable=W0212
        if len(model._modules[module_name]._modules) > 0:  # pylint: disable=W0212
            fuse_bn_recursively(model._modules[module_name])  # pylint: disable=W0212


def load_model_weights(model, pretrained_model_path, fuse_bn=False):
    """Load model weights from checkpoint"""
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    load_partial_weights(model, state_dict)
    if fuse_bn:
        fuse_bn_recursively(model)


def map_keys(ckpt_state: OrderedDict, model_state: OrderedDict) -> dict:
    """Find longest mapping of the sequence of keys with same shape.

    Match conditions:
        * suffix of keys should match (weight, bias, etc),
        * shape of values should match,
        * mapping should be not to short.
    """

    def _prepare(data, skip_nbt):
        keys = []
        shapes = []
        for k, v in data.items():
            suffix = k.rsplit(".")[-1]
            # Skip BN num_batches_tracked
            if suffix != "num_batches_tracked" or not skip_nbt:
                keys.append(k)
                shapes.append((suffix, v.shape))
        return shapes, keys

    skip_nbt = True
    if any(x.endswith("num_batches_tracked") for x in ckpt_state):
        skip_nbt = False
    ckpt_shapes, ckpt_keys = _prepare(ckpt_state, skip_nbt)
    model_shapes, model_keys = _prepare(model_state, skip_nbt)

    len_ckpt = len(ckpt_keys)
    len_model = len(model_keys)
    size_diff = abs(len_ckpt - len_model)

    # Ignore matches shorter than min_match
    min_match = 0.5 * min(len_ckpt, len_model)

    best_mapping = {}
    best_match = 0
    for i in range(0, size_diff + 1):
        if len_ckpt > len_model:
            pairs = zip(ckpt_shapes[i:], model_shapes)
            mapping = list(zip(ckpt_keys[i:], model_keys))
        else:
            pairs = zip(ckpt_shapes, model_shapes[i:])
            mapping = list(zip(ckpt_keys, model_keys[i:]))
        # Count continuous matched pairs
        match = 0
        for ckpt_shape, model_shape in pairs:
            if ckpt_shape == model_shape:
                match += 1
            else:
                break
        if match > best_match and match > min_match:
            best_match = match
            best_mapping = dict(mapping[:match])
    return best_mapping


def load_partial_weights(model, ckpt_state):
    """Load partial weights from checkpoint"""
    new_ckpt_state_dict = {}
    LOG.info("Load partial weights")
    assert len(ckpt_state) > 0
    model_state = model.state_dict()

    # Map keys
    best_mapping = map_keys(ckpt_state, model_state)
    # Expected keys order after remapping
    keys = [best_mapping.get(k, k) for k in ckpt_state]
    for key1, key2 in best_mapping.items():
        if key1 != key2:
            ckpt_state[key2] = ckpt_state.pop(key1)
    # Restore order
    for k in keys:
        ckpt_state.move_to_end(k)

    for key in ckpt_state:
        if key not in model_state:
            LOG.info(
                "Parameter %s from the checkpoint is missing in the model state dict",
                key,
            )
            continue

        if ckpt_state[key].shape == model_state[key].shape:
            new_ckpt_state_dict[key] = copy.deepcopy(ckpt_state[key])
            LOG.debug("Successfully loaded all parameters for %s", key)
        else:
            new_ckpt_state_dict[key] = copy.deepcopy(model_state[key])
            in_shape = ckpt_state[key].shape
            out_shape = model_state[key].shape
            assert len(in_shape) == len(out_shape), (
                "Shapes doesn't match. WEIGHT_CONVERSION.md might contain "
                "instruction for fix it"
            )
            min_shape = [min(in_shape[i], out_shape[i]) for i in range(len(in_shape))]

            try:
                if len(min_shape) == 4:  # Conv2d
                    new_ckpt_state_dict[key][
                        : min_shape[0], : min_shape[1], :, :
                    ] = copy.deepcopy(ckpt_state[key])[
                        : min_shape[0], : min_shape[1], :, :
                    ]
                if len(min_shape) == 1:  # Other
                    new_ckpt_state_dict[key][: min_shape[0]] = copy.deepcopy(
                        ckpt_state[key]
                    )[: min_shape[0]]
                LOG.info("Successfully loaded part of the parameters for %s", key)
            except Exception as ex:
                LOG.error(
                    "Unsuccessful trial to partially load parameters for %s %s",
                    key,
                    repr(ex),
                )

    for key in model_state:
        if key not in new_ckpt_state_dict:
            LOG.warning("Using initial model parameters for %s", key)
            new_ckpt_state_dict[key] = copy.deepcopy(model_state[key])

    model.load_state_dict(new_ckpt_state_dict)
