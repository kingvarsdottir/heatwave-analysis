import heatwave_analysis as hw
import numpy as np
import gzip

def index_fpath(index):
    return f'/home/users/kaitlyni/HW index analysis/Met Office HW index data/HadEX3_81-10_TX90_{index}_ANN.nc.gz'

start_year_int = 1980
end_year_int = 2018

# Eurasia location
lat_max, lat_min = 75, 20
lon_max, lon_min = 150, 345

index = 'HWF'
eof_num = 1

with gzip.open(index_fpath(index)) as f:
    file = hw.open_file(f)
    data = hw.process_data(file, f'TX90_{index}', start_year_int, end_year_int, lon_min, lon_max, lat_min, lat_max)
    data_detrend = hw.detrend(data, start_year_int, end_year_int, min_n = 30)

    # Add weights so higher latitudes aren't overly represented
    coslat = np.cos(np.deg2rad(data.latitude))
    wgts = np.sqrt(coslat).broadcast_like(data.isel(time = 0)) # broadcast weights array against data array so Eof() knows that weights are latitude values

    eof, pc, var_frac = hw.eof_analysis(data, wgts, eof_num)


