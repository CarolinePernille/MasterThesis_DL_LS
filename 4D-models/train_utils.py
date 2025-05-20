import torch
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import torchmetrics



def augment_4d(image, seed=None):
    """ 
    Applies simple data augmentation to a 4D image tensor.

    Augmentations include random flips along temporal and spatial axes,
    and additive Gaussian noise. Intended for use during training.

    Parameters:
    - image: 4D tensor with shape [time, x, y, z]
    - seed: Seed for reproducibility of augmentations. Used for making sure both the input and label have the same augmentation.

    Returns:
    - image: Augmented 4D tensor with shape [time, x, y, z]
    """
    random.seed(seed)

    # Temporal flip
    if random.random() > 0.5:
        image = torch.flip(image, dims=[0])

    # Spatial flip
    if random.random() > 0.5:
        image = torch.flip(image, dims=[1])
    if random.random() > 0.5:
        image = torch.flip(image, dims=[2])
    if random.random() > 0.5:
        image = torch.flip(image, dims=[3])

    # Adding Gaussian noise
    if random.random() > 0.5:
        noise = torch.rand_like(image)*0.01
        image = image + noise

    return image

def augmented_train(optimizer, model, criterion, train_loader, val_loader):
    """ 
    Trains a 4D-UNet model for one epoch and computes training and validation loss.
    Data augmentation is included in the training function.

    Parameters:
    - optimizer: Optimizer used to update model parameters.
    - model: The 4D-UNet model to be trained.
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
        data = data.squeeze(0,1)  # Removing batch size and channels dim for augmentation, shape (52, 64, 64, 64)
        labels = labels.squeeze(0,1)  

        # Setting seed for data augmentation
        seed = random.randint(0,2**32)

        # Both input and label are augmented with the same seed
        data_input = augment_4d(data, seed)
        label_input = augment_4d(labels, seed)

        data_input = data_input.unsqueeze(0).unsqueeze(0) # Add batch size and channels dim back
        label_input = label_input.unsqueeze(0).unsqueeze(0)

        data_input = data_input.to(device=device, dtype = torch.float32) 
        labels = label_input.to(device=device)
        
        optimizer.zero_grad()
        
        outputs = model(data_input)

        # Deleting variables to free up memory
        del data_input

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
            loss = criterion(outputs, labels)
            loss_val += loss.item()
        
    loss_val = loss_val/n_batch_val

    return loss_train, loss_val



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


def SSIM_loss(pred, true):
    """ 
    Computes SSIM loss.

    Parameters:
    - pred: Prediction from 4D-UNet
    - true: True label of the prediction.

    Returns:
    - 1 minus the calculated SSIM value. 
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.).to(device=device)
    ssim_value = ssim(pred, true)

    return 1. - ssim_value

def MAE_SSIM_loss(pred, true):
    """ 
    Computes the combined loss of MAE and SSIM.

    Parameters:
    - pred: Prediction from 4D-UNet
    - true: True label of the prediction.

    Returns:
    - combined_loss: A weighted sum of the two calculated loss values. Mutliplied by 100 to create bigger gradient computations.
    """
    mae_loss_fn = torch.nn.L1Loss()
    mae_loss = mae_loss_fn(pred, true)
    pred = pred.squeeze(0,1)
    true = true.squeeze(0,1)
    ssim_loss = SSIM_loss(pred, true)

    weight = 0.84
    combined_loss = (1-weight)*ssim_loss + weight*mae_loss

    return combined_loss*100

def MSE_loss(pred, true):
    """ 
    Computes the MSE loss.

    Parameters:
    - pred: Prediction from 4D-UNet
    - true: True label of the prediction.

    Returns:
    - mse: Calculated MSE loss. Mutliplied by 100 to create bigger gradient computations.
    """
    mse_function = torch.nn.MSELoss()
    mse = mse_function(pred, true)

    return mse*100

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
    labels_list = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            output = model(data)
            predictions.append(output)
            labels_list.append(labels)


    return predictions, labels_list

class CreateDataset(Dataset):
    """
    Creating a dataset consisting of inputs and labels.
    """
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return(len(self.inputs))

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]
        return x, y
    
