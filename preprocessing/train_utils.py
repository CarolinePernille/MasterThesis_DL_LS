
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from datetime import datetime

# Performance measure
from ignite.metrics import SSIM

def plot_losses(trainloss, valloss, n_epochs):
    fig, ax = plt.subplots()

    x_values = range(1,n_epochs+1)

    ax.plot(x_values, trainloss, label = 'Training loss')
    ax.plot(x_values, valloss, label = 'Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    #plt.show()

    
def predictions(model, loader):
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


def calc_SSIM(preds, true, range):
    preds_cpu = preds.cpu()
    true_cpu = true.cpu() #[0]
    print(preds_cpu.shape, true_cpu.shape)
    metric = SSIM(data_range=range)
    metric.update(np.squeeze(preds_cpu[:,0,:,:,:]),true_cpu[:,:,:,:])
    ssim_value = metric.compute()
    return ssim_value



def SSIM_loss(preds, true, ranges):
    metric = SSIM(data_range=ranges)
    metric.update((np.squeeze(preds[2:],0),true[1:]))
    ssim_value = metric.compute()
    return 1. - ssim_value

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def train(optimizer, model, criterion, train_loader, val_loader):
    n_batch = len(train_loader)
    n_batch_val = len(val_loader)
    optimizer.zero_grad(set_to_none=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    loss_val = 0.0
    loss_train = 0.0

    for data, labels in train_loader:
        model = model.to(device = device)
        model.train()
        data = data.to(device=device, dtype = torch.float32) #float32
        labels = labels.to(device=device)
        
        outputs = model(data)
        
        del data
        outputs = outputs.squeeze(1)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        loss_train += loss.item()

    loss_train = loss_train/n_batch

    model.eval()
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device=device, dtype = torch.float32) #float32
            labels = labels.to(device=device)

            outputs = model(data)
            del data
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss_val += loss.item()
        
    loss_val = loss_val/n_batch_val

    return loss_train, loss_val
    

def train_old(n_epochs, optimizer, model, criterion, train_loader, val_loader=None):
    n_batch = len(train_loader)
    losses_train = []
    optimizer.zero_grad(set_to_none=True)

    if val_loader != None:
        n_batch_val = len(val_loader)
        losses_val = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(1, n_epochs+1):
        loss_val = 0.0
        loss_train = 0.0
  
        for data, labels in train_loader:
            model = model.to(device = device)
            model.train()
            data = data.to(device=device, dtype = torch.float32) #float32
            labels = labels.to(device=device)
            
            outputs = model(data)
            
            del data
            outputs = outputs.squeeze(1)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            loss_train += loss.item()

        losses_train.append(loss_train/n_batch)

        model.eval()
        with torch.no_grad():
            if val_loader != None:
                for data, labels in val_loader:
                    data = data.to(device=device, dtype = torch.float32) #float32
                    labels = labels.to(device=device)

                    outputs = model(data)
                    del data
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)
                    loss_val += loss.item()
                avg_val_loss = loss_val/n_batch_val
                losses_val.append(avg_val_loss)
                #scheduler.step(avg_val_loss)

        

    if val_loader != None:
        return losses_train, losses_val
    else:
        return losses_train