import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import Dataset


def plot_losses(trainloss, valloss, n_epochs):
    """ 
    Plotting losses after training.

    Parameters:
    - trainloss: Training loss
    - valloss: Validation loss
    - n_epochs: Number of epochs, corresponding to the length of the loss-lists

    Returns:
    None
    """
    fig, ax = plt.subplots()

    x_values = range(1,n_epochs+1)

    ax.plot(x_values, trainloss, label = 'Training loss')
    ax.plot(x_values, valloss, label = 'Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()


    
def predictions(model, loader):
    """ 
    Get predictions from a trained model.

    Parameters:
    - model: Trained model
    - loader: Data loader for predictions (i.e validation loader for predictions on validation data)

    Returns:
    - predictions: A list containing predictions
    - true_labels: A list containing the true labels corresponding to the predictions
    """
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions = []
    true_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            output = model(data)
            predictions.append(output)
            true_labels.append(labels)


    return predictions, true_labels


def train(optimizer, model, criterion, train_loader, val_loader):
    """ 
    Trains a 3D-UNet model for one epoch and computes training and validation loss.

    Parameters:
    - optimizer: Optimizer used to update model parameters.
    - model: The 3D-UNet model to be trained.
    - criterion: Loss function used to evaluate model predictions.
    - train_loader: Dataloader providing the training data.
    - val_loader: Dataloader providing the validation data.

    Returns:
    - loss_train: Average training loss over all training batches.
    - loss_val: Average validation loss over all validation batches.
    """

    n_batch = len(train_loader)
    n_batch_val = len(val_loader)
    optimizer.zero_grad(set_to_none=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Defining variables to keep track of loss
    loss_val = 0.0
    loss_train = 0.0

    for data, labels in train_loader:
        model = model.to(device = device)
        model.train()
        data = data.to(device=device, dtype = torch.float32) 
        labels = labels.to(device=device)
        
        outputs = model(data)
        
        # Deleting variables to free up memory
        del data
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        loss_train += loss.item()

    loss_train = loss_train/n_batch

    # Setting model to evaluation mode to compute validation loss
    model.eval()
    with torch.no_grad(): # Disable gradient computation
        for data, labels in val_loader:
            data = data.to(device=device, dtype = torch.float32) 
            labels = labels.to(device=device)

            outputs = model(data)
            del data
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss_val += loss.item()
        
    loss_val = loss_val/n_batch_val

    return loss_train, loss_val
    
    
class LoadData(Dataset):

    """ 
    Custom Dataset for loading preprocessed k-space and image pairs.

    This dataset assumes that each data point has been saved as a pair of '.pt' files:
    - 'kspace{idx}.pt' in 'kspace_folder'
    - 'img{idx}.pt' in 'image_folder'

    Indexing follows:
    - Training set: indexes 0 to 363
    - Validation/Test set: indexes 0 to 51

    Parameters:
    - kspace_folder: Path to directory containing k-space files.
    - image_folder: Path to directory containing image files.
    - datatype: Type of dataset: either 'train', 'val', or 'test'.
    """

    def __init__(self, kspace_folder, image_folder, datatype):
        self.kspace_folder = kspace_folder
        self.image_folder = image_folder
        self.datatype = datatype

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        if self.datatype == 'train':
            return 364
        else:
            return 52
        
    def __getitem__(self, idx):
        """ 
        Loads and returns a single sample (k-space, image) pair.
        """
        image_path = os.path.join(self.image_folder, f'img{idx}.pt')
        kspace_path = os.path.join(self.kspace_folder, f'kspace{idx}.pt')
        image = torch.load(image_path)
        kspace = torch.load(kspace_path)
        return kspace, image