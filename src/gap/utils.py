import matplotlib
matplotlib.use('Agg')
import pylab as plt
from pySankey.sankey import sankey
from importlib import resources

BASE = resources.files(__package__.split(".")[0])
config_dir = f'{BASE}/config'
