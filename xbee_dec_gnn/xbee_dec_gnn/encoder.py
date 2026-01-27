# wire_encoder.py
from __future__ import annotations

import pickle
import zlib
from typing import Any, Dict
import numpy as np
import torch

# Header byte layout:
# bit7: TYPE (0=MP, 1=pooling)   <-- your required "first bit"
# bit6: ZLIB (payload compressed)
# bit0-5: pickle protocol (0..63)  (we use 5)
BIT_TYPE = 0x80
BIT_ZLIB = 0x40
PROTO = 5  # compact + supports protocol 5 framing (Python 3.8+)


def encode_msg(msg: Dict[str, Any], *, compress: bool = True, compress_min_bytes: int = 200) -> bytes:
    """
    Returns a single bytes payload: [1-byte header][pickle payload (maybe zlib-compressed)].

    - First bit in header is the message type:
        0 => MP
        1 => pooling
    """
    mtype = msg.get("t")
    if mtype in ("MP", "mp"):
        header = 0  # first bit = 0
    elif mtype == "pooling":
        header = BIT_TYPE  # first bit = 1
    else:
        raise ValueError(f"unknown msg type: {mtype!r}")

    payload = pickle.dumps(msg, protocol=PROTO)

    # Optional compression if it reduces size
    if compress and len(payload) >= compress_min_bytes:
        c = zlib.compress(payload, level=9)
        if len(c) < len(payload):
            payload = c
            header |= BIT_ZLIB

    # Store protocol in low bits (handy for debugging / future changes)
    header |= (PROTO & 0x3F)
    return bytes([header]) + payload


def decode_msg(data: bytes) -> Dict[str, Any]:
    """Inverse of encode_msg(). Validates first-bit type vs decoded dict['type']."""
    if not data:
        raise ValueError("empty payload")

    header = data[0]
    type_bit = 1 if (header & BIT_TYPE) else 0
    is_zlib = True if (header & BIT_ZLIB) else False
    proto = header & 0x3F  # not strictly needed; pickle can auto-detect

    payload = data[1:]
    if is_zlib:
        payload = zlib.decompress(payload)

    msg = pickle.loads(payload)

    # Sanity-check: ensure header bit matches dict "type"
    mtype = msg.get("t")
    if type_bit == 0 and mtype not in ("MP", "mp"):
        raise ValueError(f"type mismatch: header says MP, payload says {mtype!r}")
    if type_bit == 1 and mtype != "pooling":
        raise ValueError(f"type mismatch: header says pooling, payload says {mtype!r}")

    return msg


def pack_tensor(t: torch.Tensor) -> tuple[bytes, list[int]]:
    arr = t.detach().cpu().numpy().astype("<f4", copy=False)
    return arr.tobytes(order="C"), list(arr.shape)

def unpack_tensor(b: bytes, shape: list[int]) -> torch.Tensor:
    arr = np.frombuffer(b, dtype="<f4").reshape(shape)
    return torch.from_numpy(arr.copy())  # copy to own memory