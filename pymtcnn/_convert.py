"""
Convert OpenFace 2.2.0's MTCNN weights (PNet.dat, RNet.dat, ONet.dat) into the
ONNX and Core ML models that pymtcnn runs.

pymtcnn 1.1.5 and earlier shipped these models inside the package. They were
exported from a PyTorch re-implementation of OpenFace's MTCNN (ONNX: PyTorch
2.9.1, opset 11; Core ML: coremltools 8.3.0, ML Program, FP32). This module
rebuilds exactly the same files, node for node, from OpenFace's original
weights on the user's computer, so nothing derived from OpenFace has to be
distributed with pymtcnn.

Only numpy and the Python standard library are needed: the protobuf messages
and the Core ML weight file are written directly. The output does not depend
on the installed onnx, onnxruntime, protobuf or coremltools versions, and it is
byte-identical to the files in pymtcnn 1.1.5 (see EXPECTED_SHA256). The
producer metadata inside the files ("pytorch 2.9.1", "coremltools 8.3.0") is
kept so that the bytes match 1.1.5 exactly.
"""

import hashlib
import json
import os
import struct

import numpy as np

# Bump this whenever the conversion output changes; it names the cache folder.
CONVERTER_VERSION = "1"

# SHA-256 of every file this converter writes. They equal the model files that
# pymtcnn 1.1.5 shipped, so a successful conversion reproduces 1.1.5 exactly.
EXPECTED_SHA256 = {
    "pnet.onnx": "66dcf1eb5631b374384d656d6607a9ee209d42e7d316e4ae7db7e1e20fa99054",
    "rnet.onnx": "d3315a6a230b00cd1e68cb2c5ce1b90a777d753b08af6c79176df96e81f06bc5",
    "onet.onnx": "dfd5191d63f19a0d9b79552ceb2936dc97cbe1e0aaafe17a0c1737e31f57d40d",
    "pnet_fp32.mlpackage/Manifest.json": "a0066f93783393e6f5d4c8216710736ad31ae957f10ff282693f7261765ba965",
    "pnet_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel": "248b562ff58751608f5674d643b9f775d5b312090ced99b7009f901caaa821bf",
    "pnet_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin": "f802001e1752c3f8a56c54937968b733a844e6d8c179c1df1c925d7e51038fec",
    "rnet_fp32.mlpackage/Manifest.json": "d92fe28ad2ffc9b10d9adbd74c0bf4f7fa33f2aaa6c449a1a643a3fc8338d090",
    "rnet_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel": "2230957c01557b5c734cafcc1b762e34440fc8dcf57b1f2a42aabad276995a08",
    "rnet_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin": "83a163eb7a65da57b28461668fe86d5297235d801df27831e8f0be1a3d23a0d6",
    "onet_fp32.mlpackage/Manifest.json": "9309411909e245f702bdf69ebc5b619e05f611b4a854d68e8479d7fd0fc75031",
    "onet_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel": "2d34d76b41f0835c9f14ee9bd9a70dd757f9ba00bac3e97c0f9dcbd4f8a84802",
    "onet_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin": "da65e7b5d8c0dd9f4eacd19d1f0440610366db168dad8a9da641da02a9887cc4",
}

# Size in bytes of every file this converter writes (used for quick readiness checks).
EXPECTED_SIZE = {
    "pnet.onnx": 28795,
    "rnet.onnx": 404287,
    "onet.onnx": 1560547,
    "pnet_fp32.mlpackage/Manifest.json": 617,
    "pnet_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel": 7611,
    "pnet_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin": 27264,
    "rnet_fp32.mlpackage/Manifest.json": 617,
    "rnet_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel": 9177,
    "rnet_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin": 401664,
    "onet_fp32.mlpackage/Manifest.json": 617,
    "onet_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel": 11739,
    "onet_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin": 1557312,
}

# OpenFace file name for each network.
DAT_FILES = {"pnet": "PNet.dat", "rnet": "RNet.dat", "onet": "ONet.dat"}


class ConversionError(RuntimeError):
    """The OpenFace weights could not be converted."""


# ---------------------------------------------------------------------------
# OpenFace CNN reader (format of LandmarkDetector::CNN::Read in OpenFace 2.2.0)
# ---------------------------------------------------------------------------

_CONV, _POOL, _FC, _PRELU, _SIGMOID = 0, 1, 2, 3, 4
_CV_32F = 5

# Layer sequence of each OpenFace network, used to validate the input files.
_EXPECTED_LAYERS = {
    "pnet": [_CONV, _PRELU, _POOL, _CONV, _PRELU, _CONV, _PRELU, _FC],
    "rnet": [_CONV, _PRELU, _POOL, _CONV, _PRELU, _POOL, _CONV, _PRELU, _FC, _PRELU, _FC],
    "onet": [_CONV, _PRELU, _POOL, _CONV, _PRELU, _POOL, _CONV, _PRELU, _POOL, _CONV, _PRELU,
             _FC, _PRELU, _FC],
}


def read_openface_cnn(path):
    """Read an OpenFace MTCNN ``.dat`` file.

    Returns a list of layers, each a dict with a ``type`` key:
    ``conv`` (``weight`` [out, in, kh, kw], ``bias`` [out]), ``pool``
    (``kernel``, ``stride``), ``fc`` (``weight`` [in, out] as stored,
    ``bias`` [out]), ``prelu`` (``slope`` [channels]) or ``sigmoid``.
    """
    with open(path, "rb") as f:
        data = f.read()
    pos = 0

    def take(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        if pos + size > len(data):
            raise ConversionError("%s is truncated" % path)
        values = struct.unpack_from(fmt, data, pos)
        pos += size
        return values

    def matrix():
        nonlocal pos
        rows, cols, cv_type = take("<iii")
        if cv_type != _CV_32F or rows < 0 or cols < 0:
            raise ConversionError("%s has an unexpected matrix header" % path)
        count = rows * cols
        if pos + 4 * count > len(data):
            raise ConversionError("%s is truncated" % path)
        values = np.frombuffer(data, dtype="<f4", count=count, offset=pos).reshape(rows, cols)
        pos += 4 * count
        return values.astype(np.float32)

    (depth,) = take("<i")
    layers = []
    for _ in range(depth):
        (layer_type,) = take("<i")
        if layer_type == _CONV:
            num_in, num_kernels = take("<ii")
            bias = np.array(take("<%df" % num_kernels), dtype=np.float32)
            # Stored as kernels[input_map][kernel], each a row-major (kh, kw) matrix.
            kernels = [[matrix() for _ in range(num_kernels)] for _ in range(num_in)]
            weight = np.stack([np.stack([kernels[i][k] for i in range(num_in)])
                               for k in range(num_kernels)])
            layers.append({"type": "conv", "weight": weight, "bias": bias})
        elif layer_type == _POOL:
            kx, ky, sx, sy = take("<iiii")
            if kx != ky or sx != sy:
                raise ConversionError("%s has a non-square pooling layer" % path)
            layers.append({"type": "pool", "kernel": kx, "stride": sx})
        elif layer_type == _FC:
            bias = matrix().reshape(-1)
            weight = matrix()
            layers.append({"type": "fc", "weight": weight, "bias": bias})
        elif layer_type == _PRELU:
            layers.append({"type": "prelu", "slope": matrix().reshape(-1)})
        elif layer_type == _SIGMOID:
            layers.append({"type": "sigmoid"})
        else:
            raise ConversionError("%s has an unknown layer type %d" % (path, layer_type))
    if pos != len(data):
        raise ConversionError("%s has %d unexpected trailing bytes" % (path, len(data) - pos))
    return layers


_TYPE_IDS = {"conv": _CONV, "pool": _POOL, "fc": _FC, "prelu": _PRELU, "sigmoid": _SIGMOID}


def _parameters(net, layers):
    """Map OpenFace layers to the parameter tensors of the exported models.

    Keys use the module index of the PyTorch export ("0.weight", "3.bias", ...)
    and "prelu<n>" for the n-th PReLU slope.
    """
    if [_TYPE_IDS[layer["type"]] for layer in layers] != _EXPECTED_LAYERS[net]:
        raise ConversionError("The %s weights do not have the expected MTCNN layers" % DAT_FILES[net])
    param_layers = [layer for layer in layers if layer["type"] in ("conv", "fc")]
    slopes = [layer["slope"] for layer in layers if layer["type"] == "prelu"]
    params = {}
    for index, layer in zip(_ARCH[net]["param_index"], param_layers):
        if layer["type"] == "conv":
            weight = layer["weight"]
        elif net == "pnet":  # PNet's last layer is a 1x1 convolution
            weight = layer["weight"].T.reshape(layer["weight"].shape[1], -1, 1, 1)
        else:
            weight = layer["weight"].T
        params["%d.weight" % index] = np.ascontiguousarray(weight, dtype=np.float32)
        params["%d.bias" % index] = np.ascontiguousarray(layer["bias"], dtype=np.float32)
    for n, slope in enumerate(slopes):
        params["prelu%d" % n] = np.ascontiguousarray(slope, dtype=np.float32)
    for key, shape in _ARCH[net]["param_shapes"].items():
        if params[key].shape != shape:
            raise ConversionError("%s: %s has shape %s, expected %s"
                                  % (DAT_FILES[net], key, params[key].shape, shape))
    return params


# ---------------------------------------------------------------------------
# Architecture of the exported networks
# ---------------------------------------------------------------------------
# Each layer entry describes one block of the exported graph. Numbers are the
# PyTorch module indices (ONNX node names) and the TorchScript value numbers
# (Core ML variable names) of the 1.1.5 export.

_ARCH = {
    "pnet": {
        "input": (1, 3, None, None), "out_channels": 6,
        "default_hw": 12, "hw_range": (12, 1000), "batch_range": (1, 1),
        "param_index": [0, 3, 5, 7],
        "param_shapes": {"0.weight": (10, 3, 3, 3), "0.bias": (10,), "3.weight": (16, 10, 3, 3),
                         "3.bias": (16,), "5.weight": (32, 16, 3, 3), "5.bias": (32,),
                         "7.weight": (6, 32, 1, 1), "7.bias": (6,), "prelu0": (10,),
                         "prelu1": (16,), "prelu2": (32,)},
        "onnx_mul": [38, 39, 40],
        "layers": [
            ("conv", 0, "x_1"),
            ("prelu", 1, (22, 24, 26, 27), "input_3"),
            ("pool", 2, (33, 34), "input_5"),
            ("conv", 3, "x_3"),
            ("prelu", 4, (51, 53, 55, 56), "input_7"),
            ("conv", 5, "x"),
            ("prelu", 6, (71, 73, 75, 76), "input_1"),
            ("conv", 7, "var_88"),
        ],
    },
    "rnet": {
        "input": (None, 3, 24, 24), "out_channels": 6,
        "default_hw": 24, "hw_range": (24, 24), "batch_range": (1, 200),
        "param_index": [0, 3, 6, 9, 12],
        "param_shapes": {"0.weight": (28, 3, 3, 3), "0.bias": (28,), "3.weight": (48, 28, 3, 3),
                         "3.bias": (48,), "6.weight": (64, 48, 2, 2), "6.bias": (64,),
                         "9.weight": (128, 576), "9.bias": (128,), "12.weight": (6, 128),
                         "12.bias": (6,), "prelu0": (28,), "prelu1": (48,), "prelu2": (64,),
                         "prelu3": (128,)},
        "onnx_mul": [59, 60, 61, 63],
        "layers": [
            ("conv", 0, "x_1"),
            ("prelu", 1, (24, 26, 28, 29), "input_3"),
            ("pool", 2, (36, 37), "input_5"),
            ("conv", 3, "x_3"),
            ("prelu", 4, (54, 56, 58, 59), "input_7"),
            ("pool", 5, (66, 67), "input_9"),
            ("conv", 6, "x_5"),
            ("prelu", 7, (84, 86, 88, 89), "x_7"),
            ("flatten", 8, 97, "input_11"),
            ("linear", 9, "x", "linear_0"),
            ("prelu", 10, (106, 108, 110, 111), "input_1"),
            ("linear", 12, "var_115", "linear_1"),
        ],
    },
    "onet": {
        "input": (None, 3, 48, 48), "out_channels": 16,
        "default_hw": 48, "hw_range": (48, 48), "batch_range": (1, 200),
        "param_index": [0, 3, 6, 9, 12, 15],
        "param_shapes": {"0.weight": (32, 3, 3, 3), "0.bias": (32,), "3.weight": (64, 32, 3, 3),
                         "3.bias": (64,), "6.weight": (64, 64, 3, 3), "6.bias": (64,),
                         "9.weight": (128, 64, 2, 2), "9.bias": (128,), "12.weight": (256, 1152),
                         "12.bias": (256,), "15.weight": (16, 256), "15.bias": (16,),
                         "prelu0": (32,), "prelu1": (64,), "prelu2": (64,), "prelu3": (128,),
                         "prelu4": (256,)},
        "onnx_mul": [71, 72, 73, 74, 76],
        "layers": [
            ("conv", 0, "x_1"),
            ("prelu", 1, (26, 28, 30, 31), "input_3"),
            ("pool", 2, (38, 39), "input_5"),
            ("conv", 3, "x_3"),
            ("prelu", 4, (56, 58, 60, 61), "input_7"),
            ("pool", 5, (68, 69), "input_9"),
            ("conv", 6, "x_5"),
            ("prelu", 7, (86, 88, 90, 91), "input_11"),
            ("pool", 8, (97, 98), "input_13"),
            ("conv", 9, "x_7"),
            ("prelu", 10, (115, 117, 119, 120), "x_9"),
            ("flatten", 11, 128, "input_15"),
            ("linear", 12, "x", "linear_0"),
            ("prelu", 13, (137, 139, 141, 142), "input_1"),
            ("linear", 15, "var_146", "linear_1"),
        ],
    },
}

# Pooling kernel/stride of each pooling layer, by module index.
_POOL_SHAPE = {("pnet", 2): (2, 2), ("rnet", 2): (3, 2), ("rnet", 5): (3, 2),
               ("onet", 2): (3, 2), ("onet", 5): (3, 2), ("onet", 8): (2, 2)}

# Core ML package item identifiers used by the 1.1.5 packages (model, weights).
_MLPACKAGE_IDS = {
    "pnet": ("2AAE155E-4A12-40CA-BD3C-F9AD824ABF6D", "887113BF-9022-4E5A-8941-8834E941B87C"),
    "rnet": ("B2A4ADBF-F82C-47D6-9578-4DC88735E767", "BC7B0A1E-ED38-41D1-A7E0-DB8D2D3D590A"),
    "onet": ("84314B8B-820E-4959-B32E-3763F386A294", "06DA3156-984E-445A-A5EB-EA7ED1E82C54"),
}


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format writer
# ---------------------------------------------------------------------------

def _varint(value):
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _key(field, wire_type):
    return _varint((field << 3) | wire_type)


def _int(field, value):
    return _key(field, 0) + _varint(value)


def _len(field, payload):
    return _key(field, 2) + _varint(len(payload)) + payload


def _str(field, text):
    return _len(field, text.encode("utf-8"))


def _f32(field, value):
    return _key(field, 5) + struct.pack("<f", value)


def _packed_varints(field, values):
    return _len(field, b"".join(_varint(int(v)) for v in values))


def _packed_f32(field, values):
    return _len(field, np.asarray(values, dtype="<f4").tobytes())


# ---------------------------------------------------------------------------
# ONNX (onnx.proto, proto2)
# ---------------------------------------------------------------------------

_ONNX_FLOAT, _ONNX_INT64 = 1, 7
_ATTR_FLOAT, _ATTR_INT, _ATTR_TENSOR, _ATTR_INTS = 1, 2, 4, 7


def _onnx_tensor(name, dims, data_type, raw):
    msg = b"".join(_int(1, d) for d in dims) + _int(2, data_type)
    if name is not None:
        msg += _str(8, name)
    return msg + _len(9, raw)


def _onnx_attr(name, value):
    if isinstance(value, float):
        return _str(1, name) + _f32(2, value) + _int(20, _ATTR_FLOAT)
    if isinstance(value, int):
        return _str(1, name) + _int(3, value) + _int(20, _ATTR_INT)
    if isinstance(value, list):
        return _str(1, name) + b"".join(_int(8, v) for v in value) + _int(20, _ATTR_INTS)
    return _str(1, name) + _len(5, value) + _int(20, _ATTR_TENSOR)  # tensor bytes


def _onnx_node(op_type, name, inputs, outputs, attrs=()):
    msg = b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs)
    msg += _str(3, name) + _str(4, op_type)
    return msg + b"".join(_len(5, _onnx_attr(k, v)) for k, v in attrs)


def _onnx_value_info(name, dims):
    shape = b""
    for d in dims:
        shape += _len(1, _str(2, d) if isinstance(d, str) else _int(1, d))
    tensor_type = _int(1, _ONNX_FLOAT) + _len(2, shape)
    return _str(1, name) + _len(2, _len(1, tensor_type))


def build_onnx(net, params):
    """Return the bytes of ``<net>.onnx`` for the given parameters."""
    arch = _ARCH[net]
    nodes, current, prelu_n = [], "input", 0
    last = len(arch["layers"]) - 1
    zero_f32 = _onnx_tensor(None, (), _ONNX_FLOAT, struct.pack("<f", 0.0))
    for n, layer in enumerate(arch["layers"]):
        kind, idx = layer[0], layer[1]
        p = "/%d/" % idx
        if kind in ("conv", "linear"):
            out = "output" if n == last else p + ("Conv" if kind == "conv" else "Gemm") + "_output_0"
            if kind == "conv":
                k = params["%d.weight" % idx].shape[2]
                attrs = [("dilations", [1, 1]), ("group", 1), ("kernel_shape", [k, k]),
                         ("pads", [0, 0, 0, 0]), ("strides", [1, 1])]
                nodes.append(_onnx_node("Conv", p + "Conv", [current, "%d.weight" % idx, "%d.bias" % idx],
                                        [out], attrs))
            else:
                attrs = [("alpha", 1.0), ("beta", 1.0), ("transB", 1)]
                nodes.append(_onnx_node("Gemm", p + "Gemm", [current, "%d.weight" % idx, "%d.bias" % idx],
                                        [out], attrs))
            current = out
        elif kind == "prelu":
            slope = "onnx::Mul_%d" % arch["onnx_mul"][prelu_n]
            prelu_n += 1
            nodes += [
                _onnx_node("Constant", p + "Constant", [], [p + "Constant_output_0"], [("value", zero_f32)]),
                _onnx_node("Less", p + "Less", [current, p + "Constant_output_0"], [p + "Less_output_0"]),
                _onnx_node("Not", p + "Not", [p + "Less_output_0"], [p + "Not_output_0"]),
                _onnx_node("Mul", p + "Mul", [current, slope], [p + "Mul_output_0"]),
                _onnx_node("Where", p + "Where", [p + "Not_output_0", current, p + "Mul_output_0"],
                           [p + "Where_output_0"]),
            ]
            current = p + "Where_output_0"
        elif kind == "pool":
            k, s = _POOL_SHAPE[(net, idx)]
            attrs = [("ceil_mode", 1), ("dilations", [1, 1]), ("kernel_shape", [k, k]),
                     ("pads", [0, 0, 0, 0]), ("strides", [s, s])]
            nodes.append(_onnx_node("MaxPool", p + "MaxPool", [current], [p + "MaxPool_output_0"], attrs))
            current = p + "MaxPool_output_0"
        elif kind == "flatten":
            zero_i64 = _onnx_tensor(None, (), _ONNX_INT64, struct.pack("<q", 0))
            minus_one = _onnx_tensor(None, (1,), _ONNX_INT64, struct.pack("<q", -1))
            nodes += [
                _onnx_node("Shape", p + "Shape", [current], [p + "Shape_output_0"]),
                _onnx_node("Constant", p + "Constant", [], [p + "Constant_output_0"], [("value", zero_i64)]),
                _onnx_node("Gather", p + "Gather", [p + "Shape_output_0", p + "Constant_output_0"],
                           [p + "Gather_output_0"], [("axis", 0)]),
                _onnx_node("Transpose", p + "Transpose", [current], [p + "Transpose_output_0"],
                           [("perm", [0, 1, 3, 2])]),
                _onnx_node("Unsqueeze", p + "Unsqueeze", [p + "Gather_output_0"], [p + "Unsqueeze_output_0"],
                           [("axes", [0])]),
                _onnx_node("Constant", p + "Constant_1", [], [p + "Constant_1_output_0"], [("value", minus_one)]),
                _onnx_node("Concat", p + "Concat", [p + "Unsqueeze_output_0", p + "Constant_1_output_0"],
                           [p + "Concat_output_0"], [("axis", 0)]),
                _onnx_node("Reshape", p + "Reshape", [p + "Transpose_output_0", p + "Concat_output_0"],
                           [p + "Reshape_output_0"]),
            ]
            current = p + "Reshape_output_0"

    initializers = []
    for idx in arch["param_index"]:
        for suffix in ("weight", "bias"):
            name = "%d.%s" % (idx, suffix)
            t = params[name]
            initializers.append(_onnx_tensor(name, t.shape, _ONNX_FLOAT, t.astype("<f4").tobytes()))
    for n, number in enumerate(arch["onnx_mul"]):
        slope = params["prelu%d" % n]
        dims = (1, slope.size) if _is_fc_prelu(net, n) else (1, slope.size, 1, 1)
        initializers.append(_onnx_tensor("onnx::Mul_%d" % number, dims, _ONNX_FLOAT,
                                         slope.astype("<f4").tobytes()))

    if net == "pnet":
        in_dims = ["batch_size", 3, "height", "width"]
        out_dims = ["batch_size", arch["out_channels"], "out_height", "out_width"]
    else:
        in_dims = ["batch_size", 3, arch["default_hw"], arch["default_hw"]]
        out_dims = ["batch_size", arch["out_channels"]]
    graph = b"".join(_len(1, node) for node in nodes) + _str(2, "main_graph")
    graph += b"".join(_len(5, t) for t in initializers)
    graph += _len(11, _onnx_value_info("input", in_dims)) + _len(12, _onnx_value_info("output", out_dims))
    return _int(1, 6) + _str(2, "pytorch") + _str(3, "2.9.1") + _len(7, graph) + _len(8, _int(2, 11))


def _is_fc_prelu(net, n):
    """True if the n-th PReLU follows a fully connected layer."""
    prelus = [layer for layer in _ARCH[net]["layers"] if layer[0] == "prelu"]
    layers = _ARCH[net]["layers"]
    position = layers.index(prelus[n])
    return any(layer[0] in ("flatten", "linear") for layer in layers[:position])


# ---------------------------------------------------------------------------
# Core ML (Model.proto + MIL.proto, proto3) and the ML Program weight file
# ---------------------------------------------------------------------------

_MIL_BOOL, _MIL_STRING, _MIL_FP32, _MIL_INT32 = 1, 2, 11, 23
_ARRAY_FLOAT32 = 65568  # ArrayFeatureType.ArrayDataType.FLOAT32
_BLOB_ALIGN = 64


class _BlobWriter:
    """Writes the Core ML ML Program weight file (MIL blob storage, version 2)."""

    def __init__(self):
        self.chunks = [b""]  # header placeholder
        self.size = _BLOB_ALIGN
        self.count = 0

    def add_float32(self, array):
        data = np.ascontiguousarray(array, dtype="<f4").tobytes()
        meta_offset = self.size
        data_offset = meta_offset + _BLOB_ALIGN
        meta = struct.pack("<IIQQ", 0xDEADBEEF, 2, len(data), data_offset).ljust(_BLOB_ALIGN, b"\0")
        padded = data.ljust(-(-len(data) // _BLOB_ALIGN) * _BLOB_ALIGN, b"\0")
        self.chunks += [meta, padded]
        self.size = data_offset + len(padded)
        self.count += 1
        return meta_offset

    def getvalue(self):
        header = struct.pack("<II", self.count, 2).ljust(_BLOB_ALIGN, b"\0")
        return header + b"".join(self.chunks[1:])


def _mil_tensor_type(dtype, dims=None):
    msg = _int(1, dtype)
    if dims is not None:
        msg += _int(2, len(dims))
        for d in dims:
            msg += _len(3, _len(2, b"") if d is None else _len(1, _int(1, d)))
    return msg


def _mil_value_type(dtype, dims=None):
    return _len(1, _mil_tensor_type(dtype, dims))


def _mil_immediate(dtype, values):
    if dtype == _MIL_STRING:
        tensor = _len(4, b"".join(_str(1, v) for v in values))
    elif dtype == _MIL_FP32:
        tensor = _len(1, _packed_f32(1, values))
    elif dtype == _MIL_INT32:
        tensor = _len(2, _packed_varints(1, values))
    elif dtype == _MIL_BOOL:
        tensor = _len(3, _packed_varints(1, [int(bool(v)) for v in values]))
    else:
        raise ValueError(dtype)
    return _len(1, tensor)


def _mil_string_value(text):
    return _len(2, _mil_value_type(_MIL_STRING)) + _len(3, _mil_immediate(_MIL_STRING, [text]))


def _mil_name_attr(name):
    return _len(5, _str(1, "name") + _len(2, _mil_string_value(name)))


def _mil_op(op_type, inputs, out_name, out_dtype, out_dims, name):
    msg = _str(1, op_type)
    for key, binding in inputs:
        msg += _len(2, _str(1, key) + _len(2, _len(1, _str(1, binding))))
    msg += _len(3, _str(1, out_name) + _len(2, _mil_value_type(out_dtype, out_dims)))
    return msg + _mil_name_attr(name)


class _MilProgram:
    def __init__(self, blobs):
        self.ops = []
        self.blobs = blobs

    def const(self, out_name, dtype, dims, values, name=None):
        """Add a const op. ``dims`` None means a scalar."""
        if name is None:
            name = "op_" + out_name[4:] if out_name.startswith("var_") else out_name
        values = np.asarray(values)
        type_msg = _mil_value_type(dtype, dims)
        if dtype == _MIL_FP32 and values.size >= 10:
            offset = self.blobs.add_float32(values)
            value = _len(2, type_msg) + _len(5, _str(1, "@model_path/weights/weight.bin") + _int(2, offset))
        else:
            value = _len(2, type_msg) + _len(3, _mil_immediate(dtype, values.reshape(-1).tolist()))
        msg = _str(1, "const") + _len(3, _str(1, out_name) + _len(2, type_msg))
        msg += _mil_name_attr(name) + _len(5, _str(1, "val") + _len(2, value))
        self.ops.append(msg)
        return out_name

    def op(self, op_type, inputs, out_name, out_dtype, out_dims, name=None):
        if name is None:
            name = "op_" + out_name[4:] if out_name.startswith("var_") else out_name
        self.ops.append(_mil_op(op_type, inputs, out_name, out_dtype, out_dims, name))
        return out_name


def build_coreml(net, params):
    """Return (model.mlmodel bytes, weight.bin bytes, Manifest.json bytes) for ``<net>_fp32.mlpackage``."""
    arch = _ARCH[net]
    blobs = _BlobWriter()
    prog = _MilProgram(blobs)

    # Weights first, in module order (bias, then weight).
    for idx in arch["param_index"]:
        bias, weight = params["%d.bias" % idx], params["%d.weight" % idx]
        prog.const("var_%d_bias" % idx, _MIL_FP32, list(bias.shape), bias)
        prog.const("var_%d_weight" % idx, _MIL_FP32, list(weight.shape), weight)

    shape = list(arch["input"])
    current, prelu_n = "input", 0
    for layer in arch["layers"]:
        kind, idx = layer[0], layer[1]
        if kind == "conv":
            out = layer[2]
            weight = params["%d.weight" % idx]
            k = weight.shape[2]
            prog.const(out + "_pad_type_0", _MIL_STRING, None, ["valid"])
            prog.const(out + "_strides_0", _MIL_INT32, [2], [1, 1])
            prog.const(out + "_pad_0", _MIL_INT32, [4], [0, 0, 0, 0])
            prog.const(out + "_dilations_0", _MIL_INT32, [2], [1, 1])
            prog.const(out + "_groups_0", _MIL_INT32, None, [1])
            shape = [shape[0], weight.shape[0]] + [None if d is None else d - k + 1 for d in shape[2:]]
            current = prog.op("conv", [("x", current), ("weight", "var_%d_weight" % idx),
                                       ("bias", "var_%d_bias" % idx), ("strides", out + "_strides_0"),
                                       ("pad_type", out + "_pad_type_0"), ("pad", out + "_pad_0"),
                                       ("dilations", out + "_dilations_0"), ("groups", out + "_groups_0")],
                              out, _MIL_FP32, shape)
        elif kind == "prelu":
            zero, cond, slope_var, mul = ("var_%d" % v for v in layer[2])
            out = layer[3]
            slope = params["prelu%d" % prelu_n]
            prelu_n += 1
            prog.const(zero + "_promoted", _MIL_FP32, None, [0.0])
            prog.op("greater_equal", [("x", current), ("y", zero + "_promoted")], cond, _MIL_BOOL, shape)
            slope_dims = [1, slope.size] if len(shape) == 2 else [1, slope.size, 1, 1]
            prog.const(slope_var, _MIL_FP32, slope_dims, slope.reshape(slope_dims))
            prog.op("mul", [("x", current), ("y", slope_var)], mul, _MIL_FP32, shape)
            current = prog.op("select", [("cond", cond), ("a", current), ("b", mul)], out, _MIL_FP32, shape,
                              name="input" if out == "input_1" else out)
        elif kind == "pool":
            kernel_var, stride_var = ("var_%d" % v for v in layer[2])
            out = layer[3]
            k, s = _POOL_SHAPE[(net, idx)]
            prog.const(kernel_var, _MIL_INT32, [2], [k, k])
            prog.const(stride_var, _MIL_INT32, [2], [s, s])
            prog.const(out + "_pad_type_0", _MIL_STRING, None, ["custom"])
            prog.const(out + "_pad_0", _MIL_INT32, [4], [0, 0, 0, 0])
            prog.const(out + "_ceil_mode_0", _MIL_BOOL, None, [True])
            shape = shape[:2] + [None if d is None else -(-(d - k) // s) + 1 for d in shape[2:]]
            current = prog.op("max_pool", [("x", current), ("kernel_sizes", kernel_var), ("strides", stride_var),
                                           ("pad_type", out + "_pad_type_0"), ("pad", out + "_pad_0"),
                                           ("ceil_mode", out + "_ceil_mode_0")],
                              out, _MIL_FP32, shape)
        elif kind == "flatten":
            perm_var, out = "var_%d" % layer[2], layer[3]
            flat = shape[1] * shape[2] * shape[3]
            prog.const(perm_var, _MIL_INT32, [4], [0, 1, 3, 2])
            prog.const("concat_0x", _MIL_INT32, [2], [-1, flat], name="concat_0x")
            prog.op("transpose", [("x", current), ("perm", perm_var)], "x_transposed", _MIL_FP32,
                    [shape[0], shape[1], shape[3], shape[2]], name="transpose_0")
            shape = [shape[0], flat]
            current = prog.op("reshape", [("x", "x_transposed"), ("shape", "concat_0x")], out, _MIL_FP32, shape)
        elif kind == "linear":
            out, name = layer[2], layer[3]
            shape = [shape[0], params["%d.weight" % idx].shape[0]]
            current = prog.op("linear", [("x", current), ("weight", "var_%d_weight" % idx),
                                         ("bias", "var_%d_bias" % idx)], out, _MIL_FP32, shape, name=name)
    output_name = current

    block = _str(2, output_name) + b"".join(_len(3, op) for op in prog.ops)
    function = _len(1, _str(1, "input") + _len(2, _mil_value_type(_MIL_FP32, list(arch["input"]))))
    function += _str(2, "CoreML6") + _len(3, _str(1, "CoreML6") + _len(2, block))

    def kv(key, value):
        return _len(1, _len(1, _mil_string_value(key)) + _len(2, _mil_string_value(value)))

    string_type = _len(1, _int(1, _MIL_STRING))
    build_info = _len(2, _len(4, _len(1, string_type) + _len(2, string_type)))
    build_info += _len(3, _len(4, kv("coremltools-version", "8.3.0")
                                + kv("coremltools-component-torch", "2.9.1")
                                + kv("coremltools-source-dialect", "TorchScript")))
    program = _int(1, 1) + _len(2, _str(1, "main") + _len(2, function))
    program += _len(4, _str(1, "buildInfo") + _len(2, build_info))

    # Model description: input, output and metadata.
    lo_b, hi_b = arch["batch_range"]
    lo_hw, hi_hw = arch["hw_range"]
    hw = arch["default_hw"]
    ranges = b"".join(_len(1, _int(1, lo) + _int(2, hi))
                      for lo, hi in ((lo_b, hi_b), (3, 3), (lo_hw, hi_hw), (lo_hw, hi_hw)))
    array_in = _packed_varints(1, [1, 3, hw, hw]) + _int(2, _ARRAY_FLOAT32) + _len(31, ranges)
    description = _len(1, _str(1, "input") + _len(3, _len(5, array_in)))
    description += _len(10, _str(1, output_name) + _len(3, _len(5, _int(2, _ARRAY_FLOAT32))))
    meta = b"".join(_len(100, _str(1, k) + _str(2, v)) for k, v in (
        ("com.github.apple.coremltools.source", "torch==2.9.1"),
        ("com.github.apple.coremltools.version", "8.3.0"),
        ("com.github.apple.coremltools.source_dialect", "TorchScript")))
    description += _len(100, meta)
    model = _int(1, 7) + _len(2, description) + _len(502, program)

    model_id, weights_id = _MLPACKAGE_IDS[net]
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            model_id: {"author": "com.apple.CoreML", "description": "CoreML Model Specification",
                       "name": "model.mlmodel", "path": "com.apple.CoreML/model.mlmodel"},
            weights_id: {"author": "com.apple.CoreML", "description": "CoreML Model Weights",
                         "name": "weights", "path": "com.apple.CoreML/weights"},
        },
        "rootModelIdentifier": model_id,
    }
    manifest_bytes = (json.dumps(manifest, indent=4, sort_keys=True) + "\n").encode("utf-8")
    return model, blobs.getvalue(), manifest_bytes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def convert(originals, out_dir):
    """Convert OpenFace's MTCNN weights into ``out_dir``.

    ``originals`` maps "pnet"/"rnet"/"onet" to the path of PNet.dat/RNet.dat/ONet.dat.
    Writes ``<net>.onnx`` and ``<net>_fp32.mlpackage`` for each network and checks
    every file against EXPECTED_SHA256. Raises ConversionError on any mismatch.
    """
    for net in ("pnet", "rnet", "onet"):
        params = _parameters(net, read_openface_cnn(originals[net]))
        _write(os.path.join(out_dir, net + ".onnx"), build_onnx(net, params))
        model, weights, manifest = build_coreml(net, params)
        package = os.path.join(out_dir, net + "_fp32.mlpackage")
        _write(os.path.join(package, "Data", "com.apple.CoreML", "model.mlmodel"), model)
        _write(os.path.join(package, "Data", "com.apple.CoreML", "weights", "weight.bin"), weights)
        _write(os.path.join(package, "Manifest.json"), manifest)
    verify(out_dir)


def verify(model_dir):
    """Check every converted file in ``model_dir`` against EXPECTED_SHA256."""
    for rel, expected in EXPECTED_SHA256.items():
        path = os.path.join(model_dir, *rel.split("/"))
        if not os.path.isfile(path):
            raise ConversionError("Missing converted model file: %s" % rel)
        if _sha256(path) != expected:
            raise ConversionError("Converted model file %s does not match pymtcnn 1.1.5" % rel)
