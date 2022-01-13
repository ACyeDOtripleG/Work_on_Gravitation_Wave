# This script generate our training data(BBH) according to mass1, mass2 
# We synchronized the time at t = 0 and sampled 1024 points from t = -0.8 to t = 0.19
# After that I multiply the strain with (10**18)/1.3836 normalize the strain

import numpy as np
from pycbc.waveform import get_td_waveform
from scipy import interpolate

low_mass = 20
high_mass = 100

def bbhinspiral(sample_rate, n_signals):
    mass1 = np.random.uniform(low_mass, high_mass, n_signals)
    mass2 = np.random.uniform(low_mass, high_mass, n_signals)
    ht = np.zeros([n_signals,sample_rate])
    
    time  = np.linspace(-0.8, 0.19, 1024)
    for i in range(n_signals):
        if mass1[i] >= mass2[i]:
            m1 = mass1[i]
            m2 = mass2[i]
        else:
            m2 = mass1[i]
            m1 = mass2[i]
        hp, hc = get_td_waveform(approximant = 'IMRPhenomPv2',
                                 mass1 = m1,
                                 mass2 = m2,
                                 spin1z = 0.0,
                                 delta_t = 1.0/16384,
                                 f_lower = 25)
        mass1[i] = m1
        mass2[i] = m2

        function = interpolate.interp1d(hp.sample_times, hp)
        ht[i] = function(time)
    ht = ht*(10**18)/2.49213
    ht = ht.astype('float64')
    ma1 = (mass1 - low_mass)/(high_mass - low_mass)
    ma2 = (mass2 - low_mass)/(high_mass - low_mass)
    return [ht, ma1, ma2]