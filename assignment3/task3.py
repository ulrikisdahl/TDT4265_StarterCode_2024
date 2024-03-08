import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10, load_cifar10_augmented
from trainer import Trainer
from trainer import compute_loss_and_accuracy


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        self.kernel_size = 3
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=self.kernel_size,
                stride=1,
                padding=2,
            ),
            #nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=2
            ),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            
            

            #conv withot pooling
            # nn.Conv2d(in_channels=256,
            #           out_channels=256,
            #           kernel_size=self.kernel_size,
            #           stride=1,
            #           padding=1),
            # nn.ReLU()


            # nn.Conv2d(
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=self.kernel_size,
            #     stride=1,
            #     padding=2
            # ),
            # #nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.MaxPool2d(
            #     kernel_size=2,
            #     stride=2
            # ),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        # self.num_output_features = 32 * 32 * 32 # 
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.flat = nn.Flatten() # 4 * 4 * 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10368, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            #nn.Linear(2000, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out = x
        expected_shape = (batch_size, self.num_classes)

        out = self.feature_extractor(out)
        #print(out.shape)
        out = self.classifier(out)

        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

        #lr = 1/4e-2

class BestModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        self.kernel_size = 3
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=self.kernel_size,
                stride=1,
                padding=2,
            ),
            #nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=2
            ),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.kernel_size,
                stride=1,
                padding=2
            ),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            #conv withot pooling
            # nn.Conv2d(in_channels=256,
            #           out_channels=256,
            #           kernel_size=self.kernel_size,
            #           stride=1,
            #           padding=1),
            # nn.ReLU()


            # nn.Conv2d(
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=self.kernel_size,
            #     stride=1,
            #     padding=2
            # ),
            # #nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.MaxPool2d(
            #     kernel_size=2,
            #     stride=2
            # ),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        # self.num_output_features = 32 * 32 * 32 # 
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.flat = nn.Flatten() # 4 * 4 * 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            #nn.Linear(2000, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out = x
        expected_shape = (batch_size, self.num_classes)

        out = self.feature_extractor(out)
        #print(out.shape)
        out = self.classifier(out)

        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()




def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 4e-2
    early_stop_count = 4
    adam = False
    #Kernel=3x3, More capacity, ADAM, batchNorm
    # dataloaders = load_cifar10(batch_size)
    # model = ExampleModel(image_channels=3, num_classes=10)
    # trainer = Trainer(
    #     batch_size, learning_rate, early_stop_count, epochs, model, dataloaders, adam
    # )

    #With augmented data, acc = 0.59
    # dataloaders = load_cifar10_augmented(batch_size)
    # model = ExampleModel(image_channels=3, num_classes=10)
    # trainer = Trainer(
    #     batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    # )
    dataloaders = load_cifar10(batch_size)
    model = BestModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders, adam
    )


    trainer.train()

    test_loader = dataloaders[-1]
    loss_criterion = torch.nn.CrossEntropyLoss()
    test_loss = compute_loss_and_accuracy(test_loader, model, loss_criterion)
    print(test_loss)

    create_plots(trainer, "task2")


if __name__ == "__main__":
    main()
