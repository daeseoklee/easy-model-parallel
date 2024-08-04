import os
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd import Function

from emp.core.distributed import receive_tensor, send_tensor


class EmptyFunction(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.input_shapes = [input.shape for input in inputs]
        ctx.input_dtypes = [input.dtype for input in inputs]
        ctx.input_devices = [input.device for input in inputs]
        return torch.zeros([])

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        grad_inputs = tuple(
            torch.zeros(shape, dtype=dtype, device=device)
            for shape, dtype, device in zip(
                ctx.input_shapes, ctx.input_dtypes, ctx.input_devices, strict=True
            )
        )
        return grad_inputs


def empty_fn(*args) -> torch.Tensor:
    return EmptyFunction.apply(*args)  # type: ignore


class EmptyModule(torch.nn.Module):
    def _iter_grad_tensors(self, x):
        if isinstance(x, torch.Tensor) and x.requires_grad:
            yield x
        elif isinstance(x, list | tuple):
            for _x in x:
                yield from self._iter_grad_tensors(_x)
        elif isinstance(x, dict):
            for _, _x in x.items():
                yield from self._iter_grad_tensors(_x)
        else:
            pass

    def get_grad_tensors(self, args: list[Any]):
        result = []
        for arg in args:
            result.extend(list(self._iter_grad_tensors(arg)))
        return result

    def forward(self, *args, **kwargs):
        tensor_list = self.get_grad_tensors(args)  # type: ignore
        return empty_fn(*tensor_list)


class ToRankFunction(Function):
    @staticmethod
    def forward(ctx, input, src_rank, dst_rank, tag, dummy):
        assert input.is_cpu

        if dist.get_rank() == src_rank:
            send_tensor(input, dst_rank, tag=tag)
            output = torch.zeros([])  # empty_fn(input)
        elif dist.get_rank() == dst_rank:
            output = receive_tensor(src_rank, tag=tag)
        else:
            output = torch.zeros([])  # empty_fn(input)

        ctx.src_rank = src_rank
        ctx.dst_rank = dst_rank
        ctx.is_src = dist.get_rank() == src_rank
        ctx.is_dst = dist.get_rank() == dst_rank
        ctx.tag = tag

        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        assert grad_output.is_cpu
        if ctx.is_src:
            tag = ctx.tag
            if os.environ.get("DEBUG_BACKWARD", False):
                print(
                    f"Rank {dist.get_rank()} receiving grad_output (shape:{ctx.input_shape})"
                    f"from rank {ctx.dst_rank} (tag={tag})"
                )
            grad_input = receive_tensor(ctx.dst_rank, tag=tag)
        elif ctx.is_dst:
            tag = ctx.tag
            if os.environ.get(
                "DEBUG_BACKWARD", False
            ):  # TODO: turn this into os env var
                shape = tuple(grad_output.shape)
                print(
                    f"Rank {dist.get_rank()} sending grad_output (shape: {shape}) to rank {ctx.src_rank} (tag={tag})"
                )
            send_tensor(grad_output, ctx.src_rank, tag=tag)
            grad_input = torch.zeros(ctx.input_shape, dtype=ctx.input_dtype)
        else:
            grad_input = torch.zeros(ctx.input_shape, dtype=ctx.input_dtype)

        return grad_input, None, None, None, None


def to_rank_fn(
    input: torch.Tensor,
    src_rank: int,
    tgt_rank: int,
    tag=0,
) -> torch.Tensor:
    dummy = torch.empty([], requires_grad=True)
    # NOTE: the dummy ensures that the output requires grad
    return ToRankFunction.apply(input, src_rank, tgt_rank, tag, dummy)  # type: ignore
