import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dataloader(rank, world_size, batch_size=32):
    """Create DataLoader with DistributedSampler"""
    # Create toy dataset
    dataset_size = 1000
    data = torch.randn(dataset_size, 10)
    targets = torch.randn(dataset_size, 1)
    dataset = TensorDataset(data, targets)

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    return dataloader


def train(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Create model
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    # Prepare data loader
    dataloader = prepare_dataloader(rank, world_size)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        # Set epoch for sampler (important for shuffling)
        dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = ddp_model(data)
            loss = criterion(outputs, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss across all batches
        avg_loss = epoch_loss / len(dataloader)

        # Only print from rank 0
        if rank == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save model from rank 0
    if rank == 0:
        torch.save(model.state_dict(), "toy_model.pth")
        print("Model saved!")

    cleanup()


def run_demo():
    world_size = torch.cuda.device_count()

    if world_size == 0:
        print("No GPUs available. Cannot run DDP.")
        return

    print(f"Training with {world_size} GPUs")
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # Note: This requires running on a machine with multiple GPUs
    # For CPU-only training, you can use "gloo" backend instead of "nccl"
    # and remove device_ids from DDP

    run_demo()
