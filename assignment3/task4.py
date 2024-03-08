import torchvision
from torch import nn
from dataloaders import load_cifar10
import utils
from torchvision.models import ResNet18_Weights
from task2 import create_plots

from trainer import Trainer

class RESNET_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
                                            # as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False

        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer


        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers

    def forward(self, x):
        x = self.model(x)
        return x

def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4


    dataloaders = load_cifar10(batch_size)
    model = RESNET_model()
    print(model)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        True
    )
    trainer.train()
    create_plots(trainer, "task4")

if __name__ == "__main__":
    main()