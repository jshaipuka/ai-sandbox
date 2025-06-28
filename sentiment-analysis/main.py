import torch
from torch import nn


def export_model(model, tensor):
    torch.onnx.export(model, tensor, "model.onnx")


def main():
    tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.ReLU()
    )
    print(tensor.shape)
    print(model)
    print(model(tensor))
    export_model(model, tensor)


if __name__ == '__main__':
    main()
