import numpy as np
import xarray as xr
from eofs.xarray import Eof
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def eof_compute(data, eof_num, weights, eof_type = 'Variance', pcscaling = 1):
    """
    Compute empirical orthogonal function (EOF) modes and corresponding principal components (PC) for data.

    Parameters
    ----------
    data: array
        data to compute EOFs for.
    eof_num: integer
        EOF mode to compute.
    weights: array
        values to weight data by (usually sqrt(cos(data.latitude))).
    
    Optional parameters
    -------------------
    eof_type: string
        how to compute EOFs (possible inputs: 'Variance', 'Correlation').
    pcscaling: integer
        how to scale PCs with respect to data (possible inputs: 0, 1)

    Returns
    -------
    eof: spatial array
        eof as calculated according to eof_type.
    pc: temporal array
        principal component of eof; temporal eof variation.
    variance_fraction: float
        total variance which eof explains as percentage.
    """

    solver = Eof(data, weights = weights)
    if eof_type == 'Correlation':
        eof = solver.eofsAsCorrelation(neofs = eof_num)[eof_num - 1]
    elif eof_type == 'Variance':
        eof = solver.eofs(neofs = eof_num)[eof_num - 1]
    else:
        print("Input 'eof_type' valid values: 'Correlation', 'Variance'")
    pc = solver.pcs(npcs = eof_num, pcscaling = pcscaling)[:, eof_num - 1]
    variance_fraction = solver.varianceFraction(neigs = eof_num)[eof_num - 1].values * 100
    
    return eof, pc, variance_fraction

def north_test(data, weights, neigs, vfscaled = False):
    """
    Docstring for north_test
    
    :param data: Description
    :param weights: Description
    :param neigs: Description
    """
    solver = Eof(data, weights = weights)
    eigenvalues = solver.eigenvalues(neigs = neigs).values
    north = solver.northTest(neigs = neigs, vfscaled = vfscaled).values
    for i in range(neigs-1):
        diff = eigenvalues[i+1] - eigenvalues[i]
        if abs(diff) > north[i]:
            print(f'Modes {i} and {i+1} are statistically distinct')
        elif abs(diff) < north[i]:
            print(f'Modes {i} and {i+1} are not statistically distinct')

def eof_plot(eof, pc, figsize, width_ratios, plot_title = False, detrend = True, spatial_plot_type = 'color', temporal_plot_type = 'bar'):
    """
    Docstring for eof_plot
    
    :param eof: Description
    :param pc: Description
    :param figsize: Description
    :param width_ratios: Description
    :param plot_title: Description
    :param detrend: Description
    :param spatial_plot_type: Description
    :param temporal_plot_type: Description
    """
    if detrend != False:
        # Define linear fit for the PC
        def linear(x, a, b):
            return a * x + b
        time = np.arange(0, end_year_int - start_year_int + 1, 1)
        parameters, covariant_matrix = curve_fit(linear, xdata = time, ydata = pc, p0=(1000, 1000))
        a_fitted = parameters[0]
        b_fitted = parameters[1]

        pc_detrend = pc - (a_fitted * time + b_fitted)

    # Plot EOF mode spatially and corresponding PC with data temporally
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(nrows = 2, ncols = 2, width_ratios = width_ratios)
    
    ax1 = fig.add_subplot(gs[:, 0], projection = ccrs.PlateCarree(central_longitude = 0))
    ax1.set_aspect(aspect = 'auto') # force the spatial plot to fit the figure's figsize

    if spatial_plot_type == 'contour':
        cs = ax1.contour(eof.longitude, eof.latitude, eof)
        ax1.clabel(cs, cs.levels, fontsize = 8)
    elif spatial_plot_type == 'colour':
        if eof_type == 'Correlation':
            pcm = eof.plot(ax = ax1, transform = ccrs.PlateCarree(), add_colorbar = False, 
                            vmin = -1, vmax = 1.1, cmap = 'RdYlBu_r')
            cbar = fig.colorbar(pcm, ax=ax1, orientation='horizontal', label = 'Correlation coefficient',
                                boundaries = np.arange(-1, 1.1, 0.1), ticks = np.arange(-1, 1.1, 0.25))
            ax1.set_title(f'EOF{eof_num} expressed as correlation (explains {variance_fraction:.2f}% of total variance)')

        else:
            pcm = eof.plot(ax = ax1, transform = ccrs.PlateCarree(), add_colorbar = False, cmap = 'RdYlBu_r')
            cbar = fig.colorbar(pcm, ax=ax1, orientation='horizontal',
                            label = 'Variability amplitude/HW days')
            ax1.set_title(f'EOF{eof_num} (explains {variance_fraction:.2f}% of total variance)')
    else:
        print("Invalid plot type. Valid options: 'contour', 'colour'.")
    
    ax1.coastlines(resolution='110m', linewidth=1)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    gl = ax1.gridlines(draw_labels = True, dms = True, x_inline = False, y_inline = False)
    gl.top_labels = False
    gl.right_labels = False
    if lon_min > lon_max:
        ax1.set_extent([lon_min - 360, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax2 = fig.add_subplot(gs[0, 1])
    if temporal_plot_type == 'line':
        ax2.plot(years, pc, label = f'PC{eof_num}')
        ax2.plot(years, linear(time, a_fitted, b_fitted), label = f'Linear trend of PC{eof_num}')
        ax2.set_title(f'{variable_name} & PC{eof_num}')
        ax2.legend()
    elif temporal_plot_type == 'bar':
        ax2.bar(years, pc.where(pc >= 0), color = 'r')
        ax2.bar(years, pc.where(pc < 0), color = 'b')
        ax2.set_title(f'PC{eof_num}')
    if show_HWF_trend == True:
        ax2.plot(years, data_temporal_std, 'k', label = f'{variable_name}')
    ax2.grid()

    ax3 = fig.add_subplot(gs[1,1])
    if temporal_plot_type == 'line':
        ax3.plot(years, pc_detrend, label = f'De-trended PC{eof_num}')
        ax3.set_title(f'{variable_name} & de-trended PC{eof_num}')
        ax3.legend()
    elif temporal_plot_type == 'bar':
        ax3.bar(years, pc_detrend.where(pc_detrend >= 0), color = 'r')
        ax3.bar(years, pc_detrend.where(pc_detrend < 0), color = 'b')
        ax3.set_title(f'De-trended PC{eof_num}')
    if show_HWF_trend == True:
        ax3.plot(years, data_temporal_std, 'k', label = f'{variable_name}')
    ax3.grid()
    
    if plot_title != False:
        fig.suptitle(plot_title, fontsize = 16)
    
    plt.tight_layout()

