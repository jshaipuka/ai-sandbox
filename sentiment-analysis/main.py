import hiddenlayer as hl
import torch
from torch.utils.tensorboard import SummaryWriter


def visualize_model(model, tensor):
    writer = SummaryWriter('runs/visualization')
    writer.add_graph(model, tensor)
    writer.close()
    print('tensorboard --logdir=runs/visualization')

def main():
    tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2),
        torch.nn.ReLU()
    )
    print(tensor.shape)
    print(model)
    print(model(tensor))
    visualize_model(model, tensor)


if __name__ == '__main__':
    main()
