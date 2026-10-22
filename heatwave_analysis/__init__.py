"""
Heatwave analysis
=================

A Python package for analysing heatwave indices and trends.
"""

# Version info
__version__ = "0.1.0"

from . import preprocess
from . import climatology_trend
from . import eof_analysis
from . import regress

open_file = preprocess.open_file
process_data = preprocess.process_data
linear = preprocess.linear
detrend = preprocess.detrend
normalize = preprocess.normalize

climatology = climatology_trend.climatology
theil_sen_slope = climatology_trend.theil_sen_slope
kendall_pvalue = climatology_trend.kendall_pvalue
trend = climatology_trend.trend

eof_compute = eof_analysis.eof_compute
north_test = eof_analysis.north_test
eof_plot = eof_analysis.eof_plot

