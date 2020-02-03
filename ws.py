import numpy as np
import obspy as ob
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
earthquake_folder = "../../"
files = listdir(earthquake_folder)
data = ob.read("../../earthquake data/IU.ANMO.00.BHZ.M.2010.058.063000.SAC")
print(data[0].stats.sac)
try:
    print(data[0].stats.sac.a);
except:
    a=2

