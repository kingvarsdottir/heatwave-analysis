import xarray as xr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def open_file(fpath):
    """
    Open file from fpath as xarray dataset.

    Parameters
    ----------
    fpath: file path
        the fpath for the file to extract data from.

    Returns
    -------
    file: xarray.dataset
        the file with data to be analysed.
    """

    file = xr.open_dataset(fpath, decode_timedelta = True)
    return file

def process_data(file, variable_name, start_year_int, end_year_int, lon_min, lon_max, lat_min, lat_max):
    """
    Process data in file to start and end as specified by time, longitude and latitude parameters.
    
    Parameters
    ----------
    file: xarray.dataset
        the file with data to be processed.
    variable_name: string
        name of variable in file to be processed as data.
    start_year_int: integer
        year to start analysis from.
    end_year_int: integer
        year to end analysis at.
    lon_min: float
        longitude to start analysis from.
    lon_max: float
        longitude to end analysis at.
    lat_min: float
        latitude to start analysis from.
    lat_max: float
        latitude to end analysis at.
    
    Returns
    -------
    data: xr.dataset
        data extracted from file as specified by bounds.
    """
    start_year = np.datetime64((start_year_int - 1970), 'Y')
    end_year = np.datetime64((end_year_int - 1970), 'Y')
    
    # Ensure time datatype is yearly np.datetime64 starting in January
    time_dtype = file.time.dtype
    if np.issubdtype(time_dtype, np.datetime64) != True:
        ### Add code to allow generalisation of intervals so not only yearly is allowed ###
        file.assign_coords(time = np.arange(start_year, end_year, dtype = 'datetime64[Y]'))
    month = file.time.dt.month[0] # assumes all month values are the same
    if month != 1:
        file = file.assign_coords(time = file.time.to_index() - pd.DateOffset(months = month - 1))

    # Limit file to only longitude, latitude and time intervals of interest
    file = file.sel(time = slice(start_year, end_year), latitude = slice(lat_min, lat_max))
    if lon_min < 0:
        print("Use longitude from 0 to 360 convention")
        return 1
    if lon_min > lon_max:
        file = xr.concat([file.sel(longitude = slice(0.0, lon_max)), file.sel(longitude = slice(lon_min, 359.9))], dim = 'longitude')
        file = file.assign_coords(longitude = ((file.longitude + 180) % 360) - 180).sortby("longitude")
    else:
        file = file.sel(longitude = slice(lon_min, lon_max))

    data = file[variable_name]
    valid_counts = data.notnull().sum(dim = "time")
    valid_mask = valid_counts > 1
    data = data.where(valid_mask) # disclude NaN and infinite values 
    
    # If index data is of numpy timedelta64 datatype, convert to integer
    data_dtype = data.dtype
    if np.issubdtype(data_dtype, np.timedelta64) == True:
        data = data.astype('timedelta64[D]') / np.timedelta64(1, 'D')

    return data

def linear(data, a, b):
    """Simple linear fit.
    
    Parameters
    ----------
    data: array
        data to fit to.
    a: float
        gradient of fit.
    b: float
        y-intercept of fit.
    
    Returns
    -------
    fit: array
        fitted line.
    """
    fit = a * data + b
    return fit

def detrend(data, start_year_int, end_year_int, min_n):
    """
    Subtract trend from data.

    Parameters
    ----------
    data: xarray.dataset
        data to de-trend.
    start_year_int: integer
        year to start de-trending from.
    end_year_int: integer
        year to end de-trending at.
    min_n: integer
        minimum number of valid values at spatial point for trend to be computed.
    
    Returns
    -------
    data_detrend: xarray.dataset
        de-trended data.
    """

    years = np.arange(start_year_int, end_year_int + 1, 1)
    parameters, cov_matrix = curve_fit(linear, years, data)
    data_detrend = data - linear(years, *parameters)
    #years = xr.DataArray(np.arange(start_year_int, end_year_int + 1), dims = "time", coords = {"time": data.time}) # require time.dtype == int for following functions
    return data_detrend

def normalize(data, dim = None):
    """Normalize data by subtracting mean and dividing by standard deviation.
    
    Parameters
    ----------
    data: xarray or numpy dataset
        data to normalize.
    
    Optional parameters
    -------------------
    dim: string
        dimension to be normalized if data is xarray dataset
    
    Returns
    -------
    data_normalized: xarray or numpy dataset
        normalized data.
    """
    if isinstance(data, np.ndarray) == True:
        mean = np.mean(data)
        std = np.std(data)
    elif isinstance(data, xr.NamedArray) == True:
        if dim == None:
            print("For xarray dataset, must input name of dimension to normalize for.")
            return 1
        mean = data.mean(dim = dim)
        std = data.std(dim = dim)
    else:
        print("Data must be a numpy or xarray array.")
        return 1
    data_normalized = (data - mean) / std
    return data_normalized
