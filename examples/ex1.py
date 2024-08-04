import torch
import torch.distributed as dist

import emp


def ex1(devices):
    # This code will be run in multi-processing manner

    F = emp.DeviceFunctor(devices)

    device0 = F.allocate_device()
    device1 = F.allocate_device()
    device2 = F.allocate_device()

    rank = dist.get_rank()

    x1 = torch.tensor([[1.0, 2], [3, 4]], requires_grad=True)
    x2 = torch.tensor([[5.0, 6], [7, 8]], requires_grad=True)
    with F.under(device1):
        x3 = F(x1)
        print(f"Rank {rank}: x3 = {x3}")
    with F.under(device2):
        x4 = F(x2)
        print(f"Rank {rank}: x4 = {x4}")
    with F.under(device0):
        op = F(lambda a, b: torch.sum((a + b) ** 2))
    y = op(x3, x4)
    print(f"Rank {rank}: y = {y}")

    y.backward()

    print(f"Rank {rank}: x1.grad = {x1.grad}, x2.grad = {x2.grad}")


if __name__ == "__main__":
    emp.launch(ex1, ["cuda:1", "cuda:2", "cuda:3"])
