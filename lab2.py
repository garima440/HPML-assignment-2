import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import time  # To measure the time
import torch.backends.cudnn as cudnn

#for images, labels in train_loader:

 #   print(f"Batch of images shape: {images.shape}")
 #   print(f"Batch of labels shape: {labels.shape}")
 #   break

# Define the Basic Block for ResNet-18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # Define two convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)
        return out


# Define the ResNet class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # First convolutional layer (3 input channels, 64 output channels)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Create the 4 groups of blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        # The first block may have a different stride
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        # The remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Instantiate the ResNet-18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# Function to calculate top-1 accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct / labels.size(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_gradients(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None)


# Main function for training with time measurement
def main():
    # Argument parser for inputs
    parser = argparse.ArgumentParser(description='PyTorch ResNet-18 Training with Time Measurement')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--use_cpu', action='store_true', help='Force use of CPU')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to the dataset')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loader workers')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd','sgd_nesterov','adagrad','adadelta','adam'], help='Optimizer choice: sgd or sgd_nesterov or adagrad or adadelta or adam')
    args = parser.parse_args()

    # Check for device
    # Set device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f'Using device: {device}')
    cudnn.benchmark = True

    # Define transformations for the CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

    # Initialize the model, optimizer, and loss function
    model = ResNet18().to(device)

    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    if args.optimizer == 'sgd':
        # Standard SGD
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'sgd_nesterov':
        # SGD with Nesterov momentum
         optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adagrad':
        # Adagrad
        optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == 'adadelta':
        # Adadelta
        optimizer = optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        # Adam
      optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print("Using Optimizer: ", args.optimizer)
    criterion = nn.CrossEntropyLoss()

    total_time = 0.0
    total_training_time_all_epochs = 0.0
    total_loss_all_epochs = 0.0
    total_correct_all_epochs = 0
    total_samples_all_epochs = 0

    # Training for 5 epochs
    for epoch in range(5):
        print(f'\nEpoch {epoch+1}/5')

        # Start timing for the entire epoch
        torch.cuda.synchronize()
        total_epoch_start = time.perf_counter()

        total_loss = 0.0
        correct = 0
        total = 0
        model.train()


        # this the latest version
        # Initialize total data loading time for the epoch
        total_data_loading_time = 0.0
        total_training_time = 0.0

        # Create an iterator for the DataLoader
        train_iter = iter(trainloader)

        # Get the number of batches
        num_batches = len(trainloader)

        for i in range(num_batches):
            torch.cuda.synchronize()
            data_load_start = time.perf_counter()

            # Load the data
            inputs, targets = next(train_iter)

            torch.cuda.synchronize()
            data_load_end = time.perf_counter()
            total_data_loading_time += data_load_end - data_load_start

            # Move data to device (not included in timing)
            inputs = inputs.to(device)
            # Start measuring training time for this batch
            torch.cuda.synchronize()
            batch_train_start = time.perf_counter()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # End measuring training time for this batch
            torch.cuda.synchronize()
            batch_train_end = time.perf_counter()
            total_training_time += batch_train_end - batch_train_start

            # Calculate per-batch loss and accuracy
            total_loss += loss.item()
            batch_accuracy = calculate_accuracy(outputs, targets)

            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)



        # End of epoch measurements
        torch.cuda.synchronize()
        total_epoch_end = time.perf_counter()
        total_epoch_time = total_epoch_end - total_epoch_start

        # Accumulate total training time for this epoch
        total_training_time_all_epochs += total_training_time

        # Accumulate total loss, accuracy, and sample count
        total_loss_all_epochs += total_loss
        total_correct_all_epochs += correct
        total_samples_all_epochs += total
        total_time += total_data_loading_time

        # (C2.1) Data-loading time for each epoch
        print(f"Total Data Loading Time for Epoch {epoch+1}: {total_data_loading_time:.3f} seconds")

        # (C2.2) Training (i.e., mini-batch calculation) time for each epoch
        print(f"Training Time for Epoch {epoch+1}: {total_training_time:.3f} seconds")

        # (C2.3) Total running time for each epoch
        print(f"Total Running Time for Epoch {epoch+1}: {total_epoch_time:.3f} seconds")
         # Print epoch summary)
        epoch_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss: {total_loss / len(trainloader):.3f}, Top-1 Accuracy: {epoch_accuracy:.2f}%')

    # After all epochs are done, calculate the averages
    average_training_time = total_training_time_all_epochs / 5
    average_loss = total_loss_all_epochs / (5 * len(trainloader))
    average_accuracy = 100. * total_correct_all_epochs / total_samples_all_epochs

    print(f"\nTotal Data Loading Time over five epochs: {total_time:.3f} seconds")
    # Print the averages
    print(f'Average Training Time over 5 epochs: {average_training_time:.3f} seconds')
    print(f'Average Loss over 5 epochs: {average_loss:.3f}')
    print(f'Average Top-1 Accuracy over 5 epochs: {average_accuracy:.2f}%')

    num_grads = count_gradients(model)
    print(f"Number of gradients: {num_grads}")

if __name__ == '__main__':
    main()
