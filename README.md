# HPML-assignment-2

## Introduction
This assignment provides experience in profiling machine learning training and inference workloads, which is crucial for improving system performance. We will work with a Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR10 dataset, using the ResNet-18 model.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- Matplotlib (for graphing results)

## Running the Assignement
## Set up singularity container
```bash
singularity exec --nv --overlay overlay-15GB-500K.ext3:rw /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
```
## Running C2. 
Use the following command structure. Adjust the parameters as needed:

```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 2 --optimizer sgd
```
### Command breakdown:

python3: This specifies the Python interpreter to use. In this case, it’s invoking Python 3, which is necessary for running Python scripts that may utilize features from this version.

lab2.py: This is the name of the Python script you are executing. In this context, it contains the implementation of the ResNet model we created for the assignment.

--use_cuda: This flag enables the use of a CUDA-capable GPU for computations. If this option is included, the program will utilize the GPU for training. If you want to run the same script in cpu use --cpu flag.

--data_path ./data: This argument specifies the path to the dataset that the script should use. The ./data indicates that the data folder is located in the current directory from which the command is being executed. This folder should contain the CIFAR10 dataset or any other relevant data needed for the model training.

--num_workers 2: This option sets the number of worker threads used for loading the data. In this case, it is set to 2, which means that two subprocesses will be used to load the data concurrently. 

--optimizer sgd: This argument specifies the optimization algorithm to be used during training. In this case, it is set to the default mentioned in the assignemnt - sgd, which stands for Stochastic Gradient Descent. 

## Running C3.
Use the following command structure. Adjust the number of workers ranging from [0, 4, 8, 12, 16....]
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 0 --optimizer sgd
```
After running your code with various worker configurations (from 0 to your desired number), you will need to manually record the total loading times for each configuration. These times will be used to generate a bar plot that visualizes the performance of different worker counts.

You can use the following Python code to create the bar plot using Matplotlib:
```python
import matplotlib.pyplot as plt

# Data: Replace the times with your recorded loading times
workers = [0, 4, 8, 12, 16]  # Number of workers used
times = [83.267, 2.715, 2.654, 2.680, 2.727]  # Total loading times in seconds

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(workers, times)

# Customize the plot
plt.title('Worker Performance')
plt.xlabel('Number of Workers')
plt.ylabel('Time (seconds)')

# Set x-axis ticks to match exactly with the worker numbers
plt.xticks(workers)

# Add value labels on top of each bar
for i, v in enumerate(times):
    plt.text(workers[i], v, f'{v:.3f}', ha='center', va='bottom')

# Find the best performance
best_time = min(times)
best_workers = [workers[i] for i, time in enumerate(times) if time == best_time]

# Add text annotation for best performance
plt.text(0.5, 0.95, f'Best performance: {best_time:.3f} seconds\nNumber of workers: {best_workers}', 
         transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

# Show the plot
plt.tight_layout()
plt.show()

# Print the best performance to the console
print(f'Best performance: {best_time:.3f} seconds')
print(f'Number of workers: {best_workers}')

```

## Running C4.

Start from C3 Code: Begin with the code from your previous experiment (C3) where you established the baseline performance using different worker configurations.

Next, Run the code using 1 worker configuration:
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 1 --optimizer sgd
```

Note: Please note that profiling is integrated directly into the Python script

## Running C5.
To train the model using a GPU, set the flag as follows replacing the number of workers and optimizer as desired:
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 8 --optimizer sgd
```
To train the model using a CPU, set the flag as follows replacing the number of workers and optimizer as desired:
```bash
python3 lab2.py --use_cpu --data_path ./data --num_workers 8 --optimizer sgd
```
Note: Please note that profiling is integrated directly into the Python script.

## Running C6.
To train the model using SGD, set the optimizer flag to sgd:
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 8 --optimizer sgd
```
To train the model using SGD-nesterov, set the optimizer flag to sgd_nesterov:
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 8 --optimizer sgd_nesterov
```

To train the model using Adagrad, set the optimizer flag to adagrad:
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 8 --optimizer adagrad
```

To train the model using Adadelta, set the optimizer flag to adadelta:
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 8 --optimizer adadelta
```

To train the model using Adam, set the optimizer flag to adam:
```bash
python3 lab2.py --use_cuda --data_path ./data --num_workers 8 --optimizer adam
```

## Running C7.
To conduct this experiment, we will need to comment out the following lines of code from lab2.py:

In BasicBlock:
```python
# self.bn1 = nn.BatchNorm2d(out_channels)
# self.bn2 = nn.BatchNorm2d(out_channels)
# nn.BatchNorm2d(self.expansion * out_channels)
```
In the forward method:
```python
# out = self.bn1(out)
# out = self.bn2(out)
```
In ResNet Class:

Comment out the following in the __init__ method:
```python
# self.bn1 = nn.BatchNorm2d(64)
```

In the forward method:
```python
# out = self.bn1(out)
```

By just commenting these lines, the rest of the model architecture (convolutional layers, pooling, fully connected layers, etc.) will remain the same, and the model will function without batch normalization. No further changes are required.

## Running Q3.
The following functions are already added to the training script to accomplish this:
```python
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_gradients(model):
    """Count the number of gradients in a model."""
    return sum(p.numel() for p in model.parameters() if p.grad is not None)
```
After initializing my model, I have added these functions to count and print the number of trainable parameters and gradients:
```python
# In the main function, after initializing the model:
model = ResNet18().to(device)
num_params = count_parameters(model)
print(f"Number of trainable parameters: {num_params}")

# ... (training code) ...

# After training (at the end of the main function):
num_grads = count_gradients(model)
print(f"Number of gradients: {num_grads}")

```







