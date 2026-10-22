import numpy as np
import scipy.stats as sp
import xarray as xr

def climatology(data, start_year_int, end_year_int):
    """
    Calculate climatology of multidimensional dataset.
    
    Parameters
    ----------
    data: xarray.dataset
        data to calculate climatology for.
    start_year_int: int
        start year of data.
    end_year_int: int
        end year of data.
    
    Returns
    -------
    climatology: xarray.dataset
        spatial map of data averaged over specified time period
    """
    climatology = data.sum(dim = 'time',
                                 skipna = True,
                                 min_count = 1).compute() / (end_year_int - start_year_int + 1)
    return climatology

def theil_sen_slope(data, time, min_n):
    """
    Compute the Theil-Sen slope for data.

    Parameters
    ----------
    data: array
        data to calculate slope for.
    time: array
        time over which to calculate slope for.
    min_n: integer
        minimum number of valid values to calculate slope.
    
    Returns
    -------
    slope: array
        Theil-Sen slope for data.
    """

    mask = np.isfinite(data) # create array of booleans with mask[i, j] = True if element[i, j] != infinity or NaN
    if mask.sum() < min_n: # ensure data has sufficient number of valid values
        return np.nan
    slope, intercept, low_slope, high_slope = sp.theilslopes(data[mask], time[mask])
    return slope

def kendall_pvalue(data, time, min_n):
    """
    Compute the Mann-Kendall statistical significance for data's trend.

    Parameters
    ----------
    data: array
        data to calculate p-value for.
    time: array
        time over which to calculate p-value for.
    min_n: integer
        minimum number of valid values to calculate p-value.
    
    Returns
    -------
    pvalue: float
        significance level of the data's trend.
    """
    mask = np.isfinite(data) # create array of booleans with mask[i, j] = True if element[i, j] != infinity or NaN
    if mask.sum() < min_n: # ensure data has sufficient number of valid values
        return np.nan
    statistic, pvalue = sp.kendalltau(data[mask], time[mask])
    return pvalue

def trend(data, start_year_int, end_year_int):
    """
    Calculate trend and associated p-values for multidimensional dataset.
    
    Parameters
    ----------
    data: xarray.dataset
        data to calculate trends and pvalues for.
    start_year_int: int
        start year of data.
    end_year_int: int
        end year of data.
    
    Returns
    -------
    trend: array
        slopes calculated at each spatial point of data using theil_sen_slope function.
    pvalue: array
        p-values corresponding to each trend value using kendall_pvalue function.
    """
    years = xr.DataArray(np.arange(start_year_int, end_year_int + 1), dims = "time", coords = {"time": data.time}) # require time.dtype == int for following functions

    # Calculate trends
    trend = xr.apply_ufunc(theil_sen_slope, data, years,
                           input_core_dims = [["time"], ["time"]], output_core_dims = [[]],
                           vectorize = True, dask = "parallelized",
                           output_dtypes = [float])
    
    # Significance test for trends
    pvalue = xr.apply_ufunc(kendall_pvalue, data, years,
                            input_core_dims = [["time"], ["time"]], output_core_dims = [[]],
                            vectorize = True, dask = "parallelized",
                            output_dtypes = [float])

    return trend, pvalue