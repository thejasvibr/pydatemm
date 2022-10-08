# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:46:26 2022

@author: theja
"""
from pydatemm.tdoa_quality import residual_tdoa_error_nongraph
import numpy as np 
import cppyy
import scipy.spatial.distance as distance
euclidean = distance.euclidean
cppyy.load_library('C:\\Users\\theja\\anaconda3\\Library\\bin\\libiomp5md.dll')
#cppyy.load_library("/home/thejasvi/anaconda3/lib/libiomp5.so")
cppyy.add_include_path('./eigen/')
cppyy.include('./sw2002_vectorbased.h')

eigen_matXd = cppyy.gbl.Eigen.MatrixXd
eigen_vecXd = cppyy.gbl.Eigen.VectorXd


def residual_tdoa_error_nongraph(d, source, array_geom, **kwargs):
    obs_d = d.copy()
    n_channels = array_geom.shape[0]
    source_array_dist = np.apply_along_axis(euclidean, 1, array_geom, source)
    # the TDOA vector measured from data
    obtained_n_tilde = source_array_dist[1:] - source_array_dist[0]
    obtained_n_tilde /= kwargs.get('c', 343.0)
    # tdoa residual 
    obs_d /= kwargs.get('c', 343.0)
    tdoa_resid = euclidean(obs_d, obtained_n_tilde)
    tdoa_resid /= np.sqrt(n_channels)
    return tdoa_resid

def make_mock_data(nmics):
    xyz_range = np.linspace(-5,5,1000)
    micxyz = np.random.choice(xyz_range, nmics*3).reshape(-1,3)
    source = np.random.choice(xyz_range, 3)
    mic_to_source_dist = np.apply_along_axis(np.linalg.norm, 1, micxyz-source)
    R_ref0 = mic_to_source_dist[1:] - mic_to_source_dist[0]
    output = np.concatenate((micxyz.flatten(), R_ref0)).flatten()
    return output, source
#%%
#np.random.seed(78464)
nmics = int(np.random.choice(np.arange(5,10),1))
tdedata, sim_source = make_mock_data(nmics)
tdedata[-(nmics-1):] += np.random.normal(0,0.05,nmics-1)
#sim_source += np.random.normal(0,0.01,3)

arraygeom = tdedata[:nmics*3].reshape(-1,3)
aa = eigen_matXd(nmics,3)
for i in range(nmics):
    for j in range(3):
        aa[i,j] = arraygeom[i,j]
source = eigen_vecXd(3)
for i in range(3):
    source[i] = sim_source[i]

d = eigen_vecXd(nmics-1)
sim_d = tdedata[-(nmics-1):]
for i in range(nmics-1):
    d[i] = sim_d[i] 
    
res = cppyy.gbl.residual_tdoa_error(d, source, aa)

res_py = residual_tdoa_error_nongraph(sim_d, sim_source, arraygeom)
assert res == res_py

in_tde = cppyy.gbl.std.vector['double'](tdedata)
output = cppyy.gbl.sw_matrix_optim(in_tde, 343.0)
new_out = output[:3]
print(output)
#assert np.allclose(sim_source,output[:3])
calcres = cppyy.gbl.residual_tdoa_error(d, cppyy.gbl.to_VXd(new_out), aa)

