import torch
import torch.distributed as dist

# Mapping between dtype names and codes
DTYPE_MAP = {
    torch.float32: 0,
    torch.float64: 1,
    torch.int32: 2,
    torch.int64: 3,
    torch.int16: 4,
    torch.int8: 5,
    torch.uint8: 6,
    torch.bool: 7,
    torch.bfloat16: 8,
    # Add more dtypes if necessary
}

# Reverse mapping from codes to dtype
REV_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}


def send_tensor(tensor, dst, tag=0, debug=False):
    """
    Sends a tensor in a shape and type-agnostic manner.

    Args:
    - tensor (torch.Tensor): Tensor to send.
    - dst (int): Destination rank.
    - tag (int): Tag for the send operation.
    """
    # Send tensor ndim
    ndim = torch.tensor([tensor.ndim], dtype=torch.int64)
    dist.send(ndim, dst=dst, tag=4 * tag)
    if debug:
        print(f"sent ndim '{ndim}' to rank {dst}")

    # Send tensor shape
    shape = torch.tensor(tensor.shape, dtype=torch.int64)
    dist.send(shape, dst=dst, tag=4 * tag + 1)
    if debug:
        print(f"sent shape: '{shape}' to rank {dst}")

    # Send tensor dtype
    dtype_code = DTYPE_MAP[tensor.dtype]
    dtype_tensor = torch.tensor([dtype_code], dtype=torch.int32)
    dist.send(dtype_tensor, dst=dst, tag=4 * tag + 2)
    if debug:
        print(f"sent dtype: '{REV_DTYPE_MAP[dtype_code]}' to rank {dst}")

    # Send tensor data
    dist.send(tensor, dst=dst, tag=4 * tag + 3)
    if debug:
        print(f"sent tensor to rank {dst}")


def receive_tensor(src, tag=0, debug=False):
    """
    Receives a tensor in a shape and type-agnostic manner.

    Args:
    - src (int): Source rank.
    - tag (int): Tag for the receive operation.

    Returns:
    - torch.Tensor: Received tensor.
    """
    # Receive tensor ndim
    ndim = torch.empty((1,), dtype=torch.int64)
    dist.recv(ndim, src=src, tag=4 * tag)
    ndim = ndim.item()
    if debug:
        print(f"received ndim '{ndim}' from rank {src}")

    # Receive tensor shape
    shape = torch.empty((ndim,), dtype=torch.int64)  # type: ignore
    dist.recv(shape, src=src, tag=4 * tag + 1)
    if debug:
        print(f"received shape '{shape}' from rank {src}")

    # Receive tensor dtype
    dtype_tensor = torch.empty((1,), dtype=torch.int32)
    dist.recv(dtype_tensor, src=src, tag=4 * tag + 2)
    dtype_code = int(dtype_tensor.item())
    dtype = REV_DTYPE_MAP[dtype_code]
    if debug:
        print(f"received dtype '{dtype}' from rank {src}")

    # Receive tensor data
    tensor = torch.empty(tuple(shape), dtype=dtype)
    dist.recv(tensor, src=src, tag=4 * tag + 3)
    if debug:
        print(f"received tensor from rank {src}")

    return tensor
