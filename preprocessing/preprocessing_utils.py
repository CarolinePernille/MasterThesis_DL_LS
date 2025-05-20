
import numpy as np
import torch


def preprocess_scale(dataset, mean, std, normalize=False, if_X = False, val=False, test=False):
    """ 
    Preprocesses a dataset by normalizing and scaling each data point, then saves the result 
    in appropriate folders for training, validation, or testing. Intended for use in 
    3D-UNet model preparation.

    Parameters:
    - dataset: Input data to be preprocessed.
    - mean: The mean value used for normalization. Should be calculated from training data.
    - std: The standard deviation used for normalization. Should be calculated from the training data.
    - normalize: If True, normalize each image using the provided mean and standard deviation.

    - if_X: If True, indicates that the input data represents model inputs (k-space).
            If False, the data is assumed to be model labels (images).
    - val: If True, the data is saved to the validation directory.
    - test: If True, the data is saved to the test directory. 

    Notes:
    -----
    - If both `val` and `test` are False, the data is assumed to be training data.
    - Normalization is performed as: (image - mean) / std
    - Scaling maps the image values to the range [0, 1] using min-max scaling.

    Returns:
    None.
            
    """

    for i, img in enumerate(dataset):
        dataset_array = np.array(img)
        del img

        dataset_array = dataset_array.astype(np.float32) 

        # If normalize = True, the data will be normalized on the mean and std.
        if normalize:
            dataset_array = (dataset_array - mean)/ std
        
        min_val = np.min(dataset_array)
        max_val = np.max(dataset_array)

        # Scaling data based on max and min value
        scaled_data = (dataset_array - min_val) / (max_val-min_val)

        # Deleting unused variables to free up memory
        del dataset_array

        # Transforming data to tensors
        tensor_data = torch.tensor(scaled_data)

        # Saving the data into correct folders based on data type
        if if_X:
            if val:
                torch.save(tensor_data, f'preprocessed_kspace_val/kspace{i}.pt')
            elif test:
                torch.save(tensor_data, f'preprocessed_kspace_test/kspace{i}.pt')
            else:
                torch.save(tensor_data, f'preprocessed_kspace_train/kspace{i}.pt')
            del tensor_data
        else:
            tensor_data = tensor_data.unsqueeze(0)

            if val:
                torch.save(tensor_data, f'preprocessed_img_val/img{i}.pt')
            elif test:
                torch.save(tensor_data, f'preprocessed_img_test/img{i}.pt')
            else:
                torch.save(tensor_data, f'preprocessed_img_train/img{i}.pt')
            del tensor_data



def calc_mean_std(dataset):
    """
    Calculating the mean and standard deviation of dataset of type Numpy array.
    """

    means = []
    stds = []
    for data in dataset:
        mean = np.mean(data)
        std = np.std(data)
        means.append(mean)
        stds.append(std)

    return np.mean(means), np.mean(stds)

def calc_mean_std_tensor(dataset):
    """
    Calculating the mean and standard deviation of dataset of type Torch tensor.
    """
    means = []
    stds = []
    for data in dataset:
        mean = torch.mean(data)
        std = torch.std(data)
        means.append(mean)
        stds.append(std)

    return torch.mean(torch.Tensor(means)), torch.mean(torch.Tensor(stds))


def preprocess_scale_epi(dataset, mean, std, normalize=False):
    """ 
    Normalizing and scaling EPI data. To be used for 4D-UNet.

    Parameters:
    - dataset: The dataset to be pre-processed.
    - mean: The mean value used for normalization. Should be calculated on the training set.
    - std: The standard deviation used for normalization. Should be calculated on the training set.
    - normalize: If True, normalize each image using the provided mean and standard deviation.

    Notes:
    -----
    - Normalization is performed as: (image - mean) / std
    - Scaling maps the image values to the range [0, 1] using min-max scaling.

    Returns:
    - scaled_data: The pre-processed data.
    """

    if normalize:
        dataset = (dataset - mean)/ std
    
    min_val = torch.min(dataset)
    max_val = torch.max(dataset)

    scaled_data = (dataset - min_val) / (max_val-min_val)

    del dataset

    return scaled_data



