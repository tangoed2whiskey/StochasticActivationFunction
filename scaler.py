import numpy as np

def rescale_array_finder(array):
    '''
    Return mean and standard deviation of an array
    '''
    return [np.mean(array), np.std(array)]

def rescale_array(vals,array):
    '''
    Rescale an array using mean and standard deviation
    '''
    mean,std=vals
    if std==0:
        return np.array(array) - mean
    else:
        return (np.array(array) - mean)/std

def inverse_rescale_array(vals,array):
    '''
    Undo scaling of array using mean and standard deviation
    '''
    mean,std=vals
    if std==0:
        return np.array(array) + mean
    else:
        return np.array(array)*std + mean

def inverse_rescale_uncert_array(vals,array):
    '''
    Undo scaling of an array of uncertainties using standard deviation
    '''
    _,std=vals
    if std==0:
        return np.array(array)
    else:
        return np.array(array)*std

def train_scaler(data):
    '''
    Create a set of scalers for given dataset
    '''
    #Transpose data
    trans_data = np.transpose(data)
    
    #Identify and remove NaNs in data
    mask = ~np.isnan(trans_data)

    #Train scalers
    scalers = []
    for ind, arr in enumerate(trans_data):
        masks = ~np.isnan(arr)
        masked_arr = arr[masks]
        scalers.append(rescale_array_finder(masked_arr))

    #Create matrix of zeros to temporarily fill data, and
    #NaNs for re-filling
    zeros = np.zeros(trans_data.shape)
    nans = np.full(trans_data.shape,np.nan)

    #Create filled version of data for scaling
    filled_data = np.where(mask,trans_data,zeros)

    #Scaled filled data
    scaled_filled = np.array([rescale_array(scalers[ind],arr)
                              for ind, arr in enumerate(filled_data) ])

    #Replace nans
    scaled_filled = np.where(mask,scaled_filled,nans)

    #Undo transpose
    scaled_filled = np.transpose(scaled_filled)

    return scalers, scaled_filled

def scale_data(data, scalers):
    '''
    Scale a dataset using provided scalers
    '''
    #Transpose data
    trans_data = np.transpose(data)
    
    mask = ~np.isnan(trans_data)

    #Create matrix of zeros to temporarily fill data, and
    #NaNs for re-filling
    zeros = np.zeros(trans_data.shape)
    nans = np.full(trans_data.shape,np.nan)

    #Create filled version of data for scaling
    filled_data = np.where(mask,trans_data,zeros)

    #Scaled filled data
    scaled_filled = np.array([rescale_array(scalers[ind],arr)
                              for ind, arr in enumerate(filled_data) ])

    #Replace nans
    scaled_filled = np.where(mask,scaled_filled,nans)

    #Undo transpose
    scaled_filled = np.transpose(scaled_filled)

    return scaled_filled

def inverse_scale_data(data, scalers):
    '''
    Undo scaling of arrays using given scalers
    '''
    #Transpose data
    trans_data = np.transpose(data)
    
    #Identify NaNs in data
    mask = ~np.isnan(trans_data)

    #Create matrix of zeros to temporarily fill data, and
    #NaNs for re-filling
    zeros = np.zeros(trans_data.shape)
    nans = np.full(trans_data.shape,np.nan)

    #Create filled version of data for scaling
    filled_data = np.where(mask,trans_data,zeros)

    #Scaled filled data
    scaled_filled = np.array([inverse_rescale_array(scalers[ind],arr)
                              for ind, arr in enumerate(filled_data) ])

    #Replace nans
    scaled_filled = np.where(mask,scaled_filled,nans)

    #Undo transpose
    scaled_filled = np.transpose(scaled_filled)

    return scaled_filled

def inverse_scale_uncertainties(data, scalers):
    '''
    Undo scaling of uncertainty arrays using given scalers
    '''
    #Transpose data
    trans_data = np.transpose(data)
    
    #Identify NaNs in data
    mask = ~np.isnan(trans_data)

    #Create matrix of zeros to temporarily fill data, and
    #NaNs for re-filling
    zeros = np.zeros(trans_data.shape)
    nans = np.full(trans_data.shape,np.nan)

    #Create filled version of data for scaling
    filled_data = np.where(mask,trans_data,zeros)

    #Scaled filled data
    scaled_filled = np.array([inverse_rescale_uncert_array(scalers[ind],arr)
                              for ind, arr in enumerate(filled_data) ])

    #Replace nans
    scaled_filled = np.where(mask,scaled_filled,nans)

    #Undo transpose
    scaled_filled = np.transpose(scaled_filled)

    return scaled_filled