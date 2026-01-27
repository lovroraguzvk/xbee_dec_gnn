from encoder import encode_msg, decode_msg
import torch, json
import numpy as np

def pack_tensor(t: torch.Tensor) -> tuple[bytes, list[int]]:
    arr = t.detach().cpu().numpy().astype("<f4", copy=False)
    return arr.tobytes(order="C"), list(arr.shape)

def unpack_tensor(b: bytes, shape: list[int]) -> torch.Tensor:
    arr = np.frombuffer(b, dtype="<f4").reshape(shape)
    return torch.from_numpy(arr.copy())  # copy to own memory

data = torch.randn(40)  # Example data tensor

msg = {
            "t": "MP",
            "s" : "2",
            "i" : 3,
            "d" : pack_tensor(data)[0],
            "ds" : pack_tensor(data)[1]
        }

encoded_msg = encode_msg(msg)
decoded_msg = decode_msg(encoded_msg)

# json_data = json.dumps(msg).encode("utf-8")

# print("JSON encoded size in bytes:", len(json_data))
print("Custom encoded size in bytes:", len(encoded_msg))

print("original data[0:5]:", data[0:5])
print("decoded data[0:5]:", unpack_tensor(decoded_msg["d"], decoded_msg["ds"])[0:5])