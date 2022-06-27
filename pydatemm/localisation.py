'''
Localisation
============
Generates 3D source positions from a vector of TDOAs. The set of TDOAs are
built from graph synthesis. 

'''

import numpy as np 
import scipy.spatial as spatial
matmul = np.matmul

def spiesberger_wahlberg_solution(array_geometry, d, **kwargs):
    '''
    Spiesberger & Wahlberg solution. Provides *two* potential solutions, the
    correct one needs to be chosen based on validity.

    Parameters
    ----------
    array_geometry: (Nmics,3) np.array
        A Nmicsx3 array with xyz coordinates of > 4 mics.
        The first mic will be taken as the reference microphone.
    d:  (Nmics-1,1) np.array
        A Nmics-1 np.array with the range_differences in metres to the source. 
        All range_differences (:math:`d_{0..N}`), are calculated by
        taking :math:`D_{i}-D_{0}`, where :math:`D` is the direct
        range from a mic to the source.
    c: float, optional
        Speed of sound in m/s. Defaults to 340 m/s.

    Returns
    -------
    s : (3,1) np.array
        The x,y,z position of the source.
    
    Notes
    -----
    * When speed of sound is homogeneous the Spiesberger-Wahlberg method
    provides two potential solutions - only one of which is the valid one.
    Here only the 'correct' solution is defined as the one resulting
    in the same/similar TDOAs as the input TDOA.

    * The first mic in this formulation must be the origin (0,0,0). If array_geometry
    doesn't have the first mic's position as 0,0,0 - this is taken care of using 
    relative subtraction and addition. 
    
    Reference
    ---------
    1. Spiesberger & Wahlberg 2002, Probability density functions for hyperbolic and isodiachronic locations, 
       JASA, 112, 3046 (2002); doi: 10.1121/1.1513648
       
    Code modified from the batracker package.
    '''
    nmics = array_geometry.shape[0]
    if nmics<=4:
        raise ValueError(f'Array with {nmics} input. Cannot provide unique solutions')

    c = kwargs.get('c', 340.0) # m/s
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
    #print(R, tau)
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

    valid_solution = choose_SW_valid_solution(s, array_geometry+mic1, d,
                                                                      **kwargs)
    return valid_solution

def euclid_dist(X,Y):
    try:
        dist = spatial.distance.euclidean(X,Y)
    except ValueError:
        dist = np.nan
    return dist
        

def choose_SW_valid_solution(sources, array_geom, rangediffs, **kwargs):
    '''
    The Spiesberger-Wahlberg 2002 method always provides 2 potential solutions.
    The authors themselves suggest comparing the observed channel 5 and 1
    time difference ():math:`\tau_{51}` ) and the predicted :math:`\tau_{51}`
    from each source to see which one is a better fit. 

    Parameters
    ----------
    sources : list
        List with 2 sources. Each source is a (3,)/(3,1) np.array
    array_geom
    rangediffs : np.array
        Range differences to reference microphone. The ref. microphone
        is assumed to be the first row of the array_geom.
    
    Returns
    -------
    valid_solution : (3)/(3,1) np.array
        The correct solution of the two potential solutions.
    '''
    source_rangediffs =  [ make_rangediff_mat(each, array_geom) for each in sources]
    tau_ch1_sources = [each[4,0] for each in source_rangediffs]
    residuals = [rangediffs[3]-tauch1 for tauch1 in tau_ch1_sources]
    # choose the source with lower rangediff residuals
    lower_error_source = np.argmin(np.abs(residuals))
    valid_solution = sources[lower_error_source]
    return valid_solution

def make_rangediff_mat(source, array_geom):
    distmat = np.apply_along_axis(euclid_dist, 1, array_geom, source)
    rangediff = np.zeros((distmat.size, distmat.size))
    for i in range(rangediff.shape[0]):
        for j in range(rangediff.shape[1]):
            rangediff[i,j] = distmat[i]-distmat[j]
    return rangediff

#%%

# if __name__ == '__main__':
#     from simdata import simulate_1source_and1reflector_general, simulate_1source_and_1reflector_3dtristar
#     audio, distmat, array_geom, (source,ref)= simulate_1source_and1reflector_general(**{'nmics':5})
    
#     d = np.array([each-distmat[0,0] for each in distmat[0,1:]])
#     source_pos = spiesberger_wahlberg_solution(array_geom,d, c=340)
#     # get the expected tdoa match from each source 
#     print(source_pos)
    
    
    #%%
   
    
    
    

    
    