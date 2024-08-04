import inspect
import os
from collections.abc import Callable
from typing import overload

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from emp.core.function import EmptyModule, to_rank_fn
from emp.core.random import seed_all


class Counter:
    def __init__(self):
        self.counter = 0

    def get_next(self):
        self.counter += 1
        return self.counter


class DeviceManagedTensor:
    def __init__(
        self,
        tensor: torch.Tensor,
        data_device: "Device",
        to_device_counter: Counter,
        is_leaf: bool = True,
    ):
        self.data_device = data_device  # device_functor.context_device
        self.is_leaf = is_leaf
        self.to_device_counter = to_device_counter

        if self.is_leaf:
            if tensor.grad_fn is not None:
                raise ValueError("Cannot create a leaf tensor with a grad_fn")

            if dist.get_rank() == self.data_device.rank:
                self.tensor = tensor.to(device=self.data_device._device)
            else:
                self.tensor = torch.zeros([])
        else:
            # This should not be reached by user directly as a result of F(tensor)
            self.tensor = tensor

    def __repr__(self):
        return (
            f"DeviceManagedTensor(this_data={self.tensor.data}, is_leaf={self.is_leaf}"
            + f", data_rank={self.data_device.rank})"
        )

    def to_device(self, tgt_device: "Device"):
        if self.data_device.rank == tgt_device.rank:
            return self
        else:
            cpu_tensor = self.tensor.cpu()
            count = self.to_device_counter.get_next()
            moved_tensor = to_rank_fn(
                cpu_tensor, self.data_device.rank, tgt_device.rank, tag=count
            )
            if dist.get_rank() == tgt_device.rank:
                moved_tensor = moved_tensor.to(device=tgt_device._device)
            return DeviceManagedTensor(
                moved_tensor, tgt_device, self.to_device_counter, is_leaf=False
            )

    def backward(self):
        if self.is_leaf:
            raise ValueError("Cannot call backward on a leaf tensor")
        self.tensor.backward()


class DeviceManagedModule:
    def __init__(
        self,
        module: nn.Module,
        data_device: "Device",
        to_device_counter: Counter,
    ):
        self.data_device = data_device  # device_functor.context_device
        self.to_device_counter = to_device_counter

        if dist.get_rank() == self.data_device.rank:
            self.module = module.to(device=self.data_device._device)
        else:
            self.module = EmptyModule()

    def parameters(self):
        if dist.get_rank() == self.data_device.rank:
            return self.module.parameters()
        else:
            return []

    def __call__(self, x: DeviceManagedTensor):
        x = x.to_device(self.data_device)

        tensor = self.module(x.tensor)
        return DeviceManagedTensor(
            tensor, self.data_device, self.to_device_counter, is_leaf=False
        )


class DeviceManagedCallable:
    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        data_device: "Device",
        to_device_counter: Counter,
    ):
        self.data_device = data_device
        self.to_device_counter = to_device_counter

        if dist.get_rank() == self.data_device.rank:
            self.fn = fn
        else:
            self.fn = EmptyModule()

    def _transform_arg(self, arg):
        # NOTE: The order of .to_device() is important for correct inter-process communication

        if isinstance(arg, DeviceManagedTensor):
            x = arg.to_device(self.data_device)
            return x.tensor
        elif isinstance(arg, torch.Tensor):
            raise ValueError("Tensor must be wrapped in DeviceManagedTensor")
        elif isinstance(arg, list):
            return [self._transform_arg(a) for i, a in enumerate(arg)]
        elif isinstance(arg, tuple):
            return tuple(self._transform_arg(a) for i, a in enumerate(arg))
        elif isinstance(arg, dict):
            keys = sorted(arg.keys())
            return {k: self._transform_arg(arg[k]) for i, k in enumerate(keys)}
        else:
            return arg

    def __call__(self, *args, **kwargs):
        new_args = []
        for arg in args:
            new_arg = self._transform_arg(arg)
            new_args.append(new_arg)

        tensor = self.fn(*new_args, **kwargs)
        return DeviceManagedTensor(
            tensor, self.data_device, self.to_device_counter, is_leaf=False
        )


class Device:
    def __init__(self, _device, rank):
        self._device = _device
        self.rank = rank


class DeviceFunctor:
    def __init__(self, _devices):
        self._devices = _devices
        self.last_allocated_rank = -1

        self.context_device: Device | None = None
        self.to_device_counter = Counter()

    def allocate_device(self):
        rank = (self.last_allocated_rank + 1) % len(self._devices)

        _device = self._devices[rank]
        device = Device(_device, rank)

        self.last_allocated_rank = rank

        return device

    def under(self, device: Device):
        self.context_device = device
        return self

    def in_context(self):
        return self.context_device is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context_device = None

    @overload
    def __call__(self, obj: torch.Tensor) -> DeviceManagedTensor: ...

    @overload
    def __call__(self, obj: nn.Module) -> DeviceManagedModule: ...

    @overload
    def __call__(self, obj: Callable[..., torch.Tensor]) -> DeviceManagedCallable: ...

    def __call__(self, obj):
        if self.context_device is None:
            raise ValueError("You must be under a device context")

        if isinstance(obj, torch.Tensor):
            # assert that obj is a leaf tensor
            if obj.grad_fn is not None:
                raise ValueError("Device functor can be applied only to leaf tensors")
            return DeviceManagedTensor(
                obj, self.context_device, self.to_device_counter, is_leaf=True
            )
        elif isinstance(obj, nn.Module):
            return DeviceManagedModule(obj, self.context_device, self.to_device_counter)
        elif callable(obj):
            return DeviceManagedCallable(
                obj, self.context_device, self.to_device_counter
            )
        else:
            raise TypeError("Object must be a tensor or a module")


def _launch_wrapper(
    rank, world_size, target, devices, seed, backend, master_addr, master_port
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    seed_all(seed)

    target(devices)


def launch(
    target,
    devices,
    seed=42,
    backend="gloo",
    master_addr="localhost",
    master_port="12355",
):
    # Check the signature of the target function
    target_args = inspect.signature(target).parameters
    assert (
        len(target_args) == 1
    ), f"The target function must have exactly 1 argument but got {len(target_args)}"

    world_size = len(devices)
    mp.spawn(  # type: ignore
        _launch_wrapper,
        args=(
            world_size,
            target,
            devices,
            seed,
            backend,
            master_addr,
            master_port,
        ),
        nprocs=world_size,
        join=True,
    )
