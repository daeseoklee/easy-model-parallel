import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import emp


class ADataset(Dataset):
    def __init__(self, xs, ys):
        assert len(xs) == len(ys)
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def collate_fn(self, batch):
        xs = [x for x, _ in batch]
        ys = [y for _, y in batch]
        return torch.stack(xs), torch.stack(ys)


class AModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(5, 16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x[:, 0]


def ex2(devices):
    rank = dist.get_rank()

    # This code will be run in multi-processing manner

    xs = torch.randn(100, 8)
    ys = torch.randn(100, 1)

    F = emp.DeviceFunctor(devices)

    device0 = F.allocate_device()
    device1 = F.allocate_device()
    device2 = F.allocate_device()

    with F.under(device1):
        f1 = F(nn.Linear(8, 2))

    with F.under(device2):
        f2 = F(nn.Linear(8, 3))

    with F.under(device0):
        cat_op = F(lambda a, b: torch.cat([a, b], dim=1))
        final_f = F(AModule())
        loss_fn = F(lambda a, b: torch.mean((a - b) ** 2))

    params = list(f1.parameters()) + list(f2.parameters()) + list(final_f.parameters())
    optimizer = Adam(params, lr=1e-3)
    # On each process, only parameters in the non-placeholder modules are passed to the optimizer

    dataset = ADataset(xs, ys)
    loader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=4)

    for x, y in tqdm(loader, desc=f"rank {rank}"):
        # Dataloaders with the same seed will produce the same batches

        with F.under(device0):
            x = F(x)
            y = F(y)

        x1 = f1(x)  # x is transferred from device 0 to device 2
        x2 = f2(x)  # x is transferred from device 0 to device 2

        x = cat_op(x1, x2)  # x1 and x2 are transferred back to device 0
        y_pred = final_f(x)
        loss = loss_fn(y_pred, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    emp.launch(ex2, ["cuda:1", "cuda:2", "cuda:3"])
