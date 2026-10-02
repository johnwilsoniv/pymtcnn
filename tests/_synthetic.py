"""Synthetic OpenFace-format MTCNN weight files (random numbers, no OpenFace data)."""

import struct

import numpy as np

# (layer, arguments) in OpenFace's layer order. conv: (in, out, k); pool: (k, s);
# fc: (in, out); prelu: channels.
ARCHITECTURES = {
    "pnet": [("conv", 3, 10, 3), ("prelu", 10), ("pool", 2, 2), ("conv", 10, 16, 3), ("prelu", 16),
             ("conv", 16, 32, 3), ("prelu", 32), ("fc", 32, 6)],
    "rnet": [("conv", 3, 28, 3), ("prelu", 28), ("pool", 3, 2), ("conv", 28, 48, 3), ("prelu", 48),
             ("pool", 3, 2), ("conv", 48, 64, 2), ("prelu", 64), ("fc", 576, 128), ("prelu", 128),
             ("fc", 128, 6)],
    "onet": [("conv", 3, 32, 3), ("prelu", 32), ("pool", 3, 2), ("conv", 32, 64, 3), ("prelu", 64),
             ("pool", 3, 2), ("conv", 64, 64, 3), ("prelu", 64), ("pool", 2, 2), ("conv", 64, 128, 2),
             ("prelu", 128), ("fc", 1152, 256), ("prelu", 256), ("fc", 256, 16)],
}
FILE_NAMES = {"pnet": "PNet.dat", "rnet": "RNet.dat", "onet": "ONet.dat"}


def _matrix(a):
    a = np.asarray(a, dtype="<f4")
    return struct.pack("<iii", a.shape[0], a.shape[1], 5) + a.tobytes()


def write_dat(path, net, seed=0):
    """Write a random-weight MTCNN file in OpenFace's binary format."""
    rng = np.random.default_rng(seed)
    layers = ARCHITECTURES[net]
    out = struct.pack("<i", len(layers))
    for layer in layers:
        kind = layer[0]
        if kind == "conv":
            _, n_in, n_out, k = layer
            out += struct.pack("<iii", 0, n_in, n_out)
            out += rng.normal(0, 0.1, n_out).astype("<f4").tobytes()
            for _ in range(n_in):
                for _ in range(n_out):
                    out += _matrix(rng.normal(0, 0.3, (k, k)))
        elif kind == "pool":
            _, k, s = layer
            out += struct.pack("<iiiii", 1, k, k, s, s)
        elif kind == "fc":
            _, n_in, n_out = layer
            out += struct.pack("<i", 2) + _matrix(rng.normal(0, 0.1, (n_out, 1)))
            out += _matrix(rng.normal(0, 0.05, (n_in, n_out)))
        elif kind == "prelu":
            out += struct.pack("<i", 3) + _matrix(rng.uniform(0.0, 0.5, (layer[1], 1)))
    with open(path, "wb") as f:
        f.write(out)
    return out


def reference_forward(net, params, x):
    """Plain numpy forward pass of the exported graph (NCHW float32)."""
    from numpy.lib.stride_tricks import sliding_window_view

    x = x.astype(np.float64)
    prelu = 0
    for layer in ARCHITECTURES[net]:
        kind = layer[0]
        if kind == "conv":
            k = layer[3]
            idx = _module_index(net, "conv", layer)
            w = params["%d.weight" % idx].astype(np.float64)
            b = params["%d.bias" % idx].astype(np.float64)
            windows = sliding_window_view(x, (k, k), axis=(2, 3))  # n, c, h, w, k, k
            x = np.einsum("nchwij,ocij->nohw", windows, w) + b[None, :, None, None]
        elif kind == "prelu":
            slope = params["prelu%d" % prelu].astype(np.float64)
            prelu += 1
            shape = (1, -1) if x.ndim == 2 else (1, -1, 1, 1)
            x = np.where(x >= 0, x, x * slope.reshape(shape))
        elif kind == "pool":
            _, k, s = layer
            n, c, h, w = x.shape
            oh, ow = -(-(h - k) // s) + 1, -(-(w - k) // s) + 1
            padded = np.full((n, c, (oh - 1) * s + k, (ow - 1) * s + k), -np.inf)
            padded[:, :, :h, :w] = x
            x = sliding_window_view(padded, (k, k), axis=(2, 3))[:, :, ::s, ::s].max(axis=(4, 5))
        elif kind == "fc":
            idx = _module_index(net, "fc", layer)
            w = params["%d.weight" % idx].astype(np.float64)
            b = params["%d.bias" % idx].astype(np.float64)
            if x.ndim == 4 and w.ndim == 4:  # PNet: 1x1 convolution
                x = np.einsum("nchw,oc->nohw", x, w[:, :, 0, 0]) + b[None, :, None, None]
            else:
                if x.ndim == 4:
                    x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], -1)
                x = x @ w.T + b
    return x


_INDEX = {"pnet": [0, 3, 5, 7], "rnet": [0, 3, 6, 9, 12], "onet": [0, 3, 6, 9, 12, 15]}


def _module_index(net, kind, layer):
    params = [entry for entry in ARCHITECTURES[net] if entry[0] in ("conv", "fc")]
    return _INDEX[net][[i for i, entry in enumerate(params) if entry is layer][0]]
