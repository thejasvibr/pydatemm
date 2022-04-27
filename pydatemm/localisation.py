'''
Localisation
============
Generates 3D source positions from a vector of TDOAs. The set of TDOAs are
built from graph synthesis. 

'''

import numpy as np 
import scipy.spatial as spatial
matmul = np.matmul
def spherical_interpolation_huang(d, **kwargs):
    '''
    

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.
    S : (Mmics, 3) np.array
        Sensory geometry with xyz coordinates
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    References
    ----------
    * Huang et al. 2001, Real-time passive source localization: a practical
    linear-correction least-squares approach, ICASSP
    '''
    



def spiesberger_wahlberg_solution(array_geometry, d, **kwargs):
    '''
    Parameters
    ----------
    array_geometry: (Nmics,3) np.array
        A Nmicsx3 array with xyz coordinates of >= 4 mics.
        The first mic will be taken as the reference microphone.
    d:  (Nmics-1,1) np.array
        A Nmics-1 np.array with the range_differences in metres to the source. 
        All range_differences (:math:`d_{0..N}`), are calculated by
        taking :math:`D_{i}-D_{0}`, where :math:`D` is the direct
        range from a mic to the source.
    Returns
    -------
    s : list
        A list with two 3x1 np.arrays with the x,y,z positions of the source.
        The two xyz positions describe two possible solutions to the given 
        array geometry and range differences.
    
    Notes
    -----
    The first mic in this formulation must be the origin (0,0,0). If array_geometry
    doesn't have the first mic's position as 0,0,0 - this is taken care of using 
    relative subtraction and addition. 
    
    Code taken from the batracker package.
    '''
    c = kwargs.get('c', 338.0) # m/s
    # check that the 1st mic is origin - else set it to 0,0,0
    if not np.array_equal(array_geometry[0,:], np.array([0,0,0])):
        mic1_notorigin = True
        mic1 = array_geometry[0,:]
        array_geometry = array_geometry - mic1
    else:
        mic1_notorigin = False
        
    # the receiver matrix- excluding the first channel.
    R = array_geometry[1:,:]
    tau = d.copy()/c # to keep symbol conventions the same
    
    try:
        R_inv = np.linalg.inv(R)
    except:
        R_inv = np.linalg.pinv(R)
    
    Nrec_minus1 = R.shape[0]
    b = np.zeros(Nrec_minus1)
    f = np.zeros(Nrec_minus1)
    g = np.zeros(Nrec_minus1)
    for i in range(Nrec_minus1):
        b[i] = np.linalg.norm(R[i,:])**2 - (c*tau[i])**2
        f[i] = (c**2)*tau[i]
        g[i] = 0.5*(c**2-c**2)
    
    
    a1 = matmul(matmul(R_inv, b).T, matmul(R_inv,b))
    a2 = matmul(matmul(R_inv, b).T, matmul(R_inv,f))
    a3 = matmul(matmul(R_inv, f).T, matmul(R_inv,f))
    
    # quadratic equation is ax^2 + bx + c = 0
    # solution is x = (-b +/- sqrt(b^2 - 4ac))/2a
    # replace 
    
    a_quad = a3 - c**2
    b_quad = -a2
    c_quad = a1/4.0
    
    t_solution1 = (-b_quad + np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)
    t_solution2 = (-b_quad - np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)
    t1 = (t_solution1 , t_solution2)
    
    
    
    s = [matmul(R_inv,b*0.5) - matmul(R_inv,f)*t1[0],
         matmul(R_inv,b*0.5) - matmul(R_inv,f)*t1[1]]
    if mic1_notorigin:
        for each in s:
            each += mic1
    return s

if __name__ == '__main__':
    from simdata import simulate_1source_and_1reflector, simulate_1source_and_1reflector_3dtristar
    audio, distmat, array_geom, (source,ref)= simulate_1source_and_1reflector()
    d = np.array([each-distmat[0,0] for each in distmat[0,1:]])
    spiesberger_wahlberg_solution(array_geom,d, c=340)

    
    