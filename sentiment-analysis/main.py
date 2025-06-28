import torch
from torch import nn


def export_model(model, tensor):
    torch.onnx.export(model, tensor, "model.onnx")


def main():
    batch = torch.tensor([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ], dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.ReLU()
    )
    print(batch.shape)
    print(model)
    print(model(batch))
    export_model(model, batch)


if __name__ == '__main__':
    main()
