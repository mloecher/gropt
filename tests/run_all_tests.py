import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Compile the library (won't run if nothing has changed)
import os
cwd = os.getcwd()
os.chdir('../python/')
import subprocess
out = subprocess.run(["python", "setup.py", "build_ext", "--inplace"], stdout=subprocess.PIPE)
os.chdir(cwd)

import sys
sys.path.append('../python/')
import gropt
from helper_utils import *

import hdf5storage
from tqdm import tqdm


def compare_waveforms(G0, G1):
    if not (G0.size == G1.size):
        print('ERROR: output waveforms are not the same size')
    
    res0 = np.linalg.norm(G0-G1) 
    res1 = np.linalg.norm(G0+G1)
    min_res = min(res0, res1)
    
    rel_err = min_res/np.linalg.norm(G0)

    if (rel_err > 1e-3):
        print('ERROR: output waveforms are different')

def run_test_diff_v1(data):
    params = data['params_in']
    if params['diffmode'] == 1:
        params['mode'] = 'diff_beta'
    elif params['diffmode'] == 2:
        params['mode'] = 'diff_bval' 
    
    if 'N' in params:
        if params['N'] > 0:
            params['N0'] = params['N']

    G, limit_break = gropt.gropt(params)

    compare_waveforms(data['G'], G)

def run_testcase(casefile):

    data = hdf5storage.read(filename=casefile)
    if data['version'] == 'diff_v1':
        run_test_diff_v1(data)


import os

all_cases = []

for root, dirs, files in os.walk('./cases/'):
    for f in files:
        if f.endswith('.h5'):
             all_cases.append(os.path.join(root, f))

for i in tqdm(range(len(all_cases)), desc='Test Progress', ncols=110):
    run_testcase(all_cases[i])