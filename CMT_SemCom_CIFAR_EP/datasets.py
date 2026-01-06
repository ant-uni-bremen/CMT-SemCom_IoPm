import torch
import os
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



def load_cifar10(path="datasets/CIFAR", seed=42):
    # set seed
    random.seed(seed)
    np.random.seed(seed)

    # load Dataser
    os.makedirs(path, exist_ok= True)
    
    cifar_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root=path, download=True, train=True, transform=cifar_transform)
    test_dataset  = torchvision.datasets.CIFAR10(root=path, download=True, train=False, transform=cifar_transform)

    train_data_tensor = torch.tensor(train_dataset.data).permute(0, 3, 1, 2).float()/255.0 # changes the order of dimensions from (N, H, W, C) to (N, C, H, W), which is the expected format for PyTorch 
    train_targets = np.array(train_dataset.targets)


    # Task1: classification of Bird
    task_one_targets = np.where(train_targets == 2, 1, 0) # creates a new array where the label is 1 if the original label is 2 (representing the "bird" class in CIFAR-10) and 0 otherwise.
    task_one_targets = torch.tensor(task_one_targets, dtype=torch.float32)
    first_task_dataset = torch.utils.data.TensorDataset(train_data_tensor, task_one_targets)

    # Task2: categorical classification
    # We may use the train_dataset itself
    task_two_targets = torch.tensor(train_targets, dtype=torch.float32)
    second_task_dataset = torch.utils.data.TensorDataset(train_data_tensor, task_two_targets)

    # ---->> Test set of MNIST has a different datatype from the training set, take care

    # Accuracy Test: Task1
    test_data_tensor = torch.tensor(test_dataset.data).permute(0, 3, 1, 2).float()/255.0
    test_targets = np.array(test_dataset.targets)
    first_test_target = np.where(test_targets == 2, 1, 0)
    first_test_target = torch.tensor(first_test_target, dtype=torch.float32)
    first_test_dataset = torch.utils.data.TensorDataset(test_data_tensor, first_test_target)

    # Accuracy Test: Task2
    second_test_target = torch.tensor(test_targets, dtype=torch.float32)
    second_test_dataset = torch.utils.data.TensorDataset(test_data_tensor, second_test_target)

    return first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset

# Why the data has been changed to numpy?! Maybe because it is for a long time ago and hasn't been updated

def load_dataset(key):
    loader = {
        "CIFAR": load_cifar10,
        #"OMNIGLOT": load_omniglot,
        #"Histopathology": load_histopathology,
        #"FreyFaces": load_freyfaces,
        #"OneHot": load_one_hot
    }
    return loader[key]()

 


# Tests and veryfing


#first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset = load_dataset(key="CIFAR")

#---->> (checking the training sets we have made)

# Convert the first_task_dataset to numpy arrays 
#sample_array, target_array = zip(*second_task_dataset)
#sample_array = torch.stack(sample_array).numpy()
#target_array = torch.stack(target_array).numpy()

# Now you can check the shape
#print("CommonInput shape:", sample_array.shape)
#print("Label shape:", target_array.shape)


#num_label_0 = np.sum(target_array == 0)

#print(f"Number of data points labeled 0: {num_label_0}")

# Assuming resampled_dataset is your resampled dataset
#N = len(second_task_dataset)

#print(f"Number of samples in the first task dataset: {N}")


# ---->> (Ploting a recovered data)

#index_2 = next(i for i, (image, label) in enumerate(second_task_dataset) if label == 6)

# Extract and plot the image
#image_2, label = second_task_dataset[index_2]
#image_2 = image_2.permute(1, 2, 0)  # Convert to NumPy array
#image_2 = (image_2 / 2) + 0.5
#image_2 = image_2.reshape(28, 28)  # Reshape to the original 2D shape
#plt.imshow(image_2)
#plt.title(f'CIFAR Image labeled as {label}')
#plt.show()


# ---->> (Testing the way we use the test dataset reshaping in the eval loop)

#test_reshaped = test_dataset.data.view(-1,784).squeeze()

#neu_test_dataset = torch.utils.data.TensorDataset(test_reshaped, test_dataset.targets)


#index_5 = next(i for i, (image, label) in enumerate(second_test_dataset) if label == 8)

# Extract and plot the image
#image_5, label = second_test_dataset[index_5]
#image_5 = image_5.numpy()  # Convert to NumPy array
#image_5 = image_5.reshape(28, 28)  # Reshape to the original 2D shape
#plt.imshow(image_5, cmap='gray')
#plt.title(f'MNIST Image labeled as {label}')
#plt.show()

# Assuming your datasets are named first_task_dataset, second_task_dataset, first_test_dataset, and second_test_dataset

# Check data type of training datasets
#print("First Task Training Dataset Data Type:", first_task_dataset.tensors[0].dtype)
#print("Second Task Training Dataset Data Type:", second_task_dataset.tensors[0].dtype)

# Check data type of testing datasets
#print("First Task Test Dataset Data Type:", first_test_dataset.tensors[0].dtype)
#print("Second Task Test Dataset Data Type:", second_test_dataset.tensors[0].dtype)