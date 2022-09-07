# -*- coding: utf-8 -*-
"""
Playing around - a CombineAll implementation 
============================================

References
----------
* Kreissig & Yang 2013, Fast & reliable ...., ICASSP 
Created on Tue Jun 14 11:59:03 2022

@author: theja
"""
import numpy as np 
from copy import deepcopy
from itertools import chain, product
import time
#%load_ext line_profiler
#%%
def get_Nvl_and_Nnotvl(Acc:np.array, V:set, l:set):
    '''Function which performs the Nvl and N_not_vl calculation
    together.
    '''
    Nvl, Nnotvl = set([]), set([])
    if len(l)<1:
        return V, Nnotvl
    else:
        for v in V:
            compatible = False
            conflict = False
            for u in l:
                if Acc[v,u]==-1:
                    conflict = True
                elif Acc[v,u]==1:
                    compatible = True
            if conflict:
                Nnotvl.add(v)
            elif compatible:
                Nvl.add(v)
                    
    return Nvl, Nnotvl

def combine_all(Acc, V, l, X):
    '''
    

    Parameters
    ----------
    Acc : (N_cfl, N_cfl) np.array
        DESCRIPTION.
    V : set
    l : set
    X : set

    Returns
    -------
    None.

    '''
    # determine N_v(l) and !N_v(l)
    # !N_v(l) are the vertices incompatible with the current solution
    #N_vl = get_Nvl(Acc, V, l)
    #N_not_vl = get_NOT_Nvl(Acc, V, l)
    N_vl, N_not_vl = get_Nvl_and_Nnotvl(Acc, V, l)
    # print("Nvl: ", N_vl)
    # print("notNvl ", N_not_vl)
    # print(f'l:{l}, X:{X}, V:{V}, N_vl:{N_vl}, N_notvl:{N_not_vl}, X:{X}')
    solutions_l = []
    if len(N_vl) == 0:
        solutions_l.append(l)
        #print(l)
    else:
        # remove conflicting neighbour
        V = V.difference(N_not_vl)
        # unvisited compatible neighbours
        Nvl_wo_X = N_vl.difference(X)
        #print(f'   Vdiff: {V}, NvlwoX: {Nvl_wo_X}')
        for n in Nvl_wo_X:
            #print(f'n: {n}')
            Vx = V.difference(set([n]))
            lx = l.union(set([n]))
            solution = combine_all(Acc, Vx, lx, X)
            if solution:
                for each in solution:
                    solutions_l.append(each)
            X = X.union(set([n]))
    return solutions_l
#%%
if __name__ == '__main__':
    A = np.array([[ 0, 1, 0, 0,-1,-1],
                  [ 1, 0, 1, 1, 0, 1],
                  [ 0, 1, 0,-1, 1, 0],
                  [ 0, 1,-1, 0,-1, 0],
                  [-1, 0, 1,-1, 0, 1],
                  [-1, 1, 0, 0, 1, 0]])
    # Now need to find a way to flatten the solution outputs.
    qq = combine_all(A, set(range(6)), set([]), set([]))
    
    #A[:,0] = 0
    #A[[1,5],0] = -1
    # qq2 = combine_all(A, set(range(6)), set([]), set([]))
    #%lprun -f combine_all combine_all(A, set(range(6)), set([]), set([]))
    # qq = combine_all(A, set([1,3,4,5,6]), set([2]), set([1]))
    # qq = combine_all(A, set([1,2,3,4,6]), set([5]), set([1,2,3,4]))
    #%%
    start = time.perf_counter_ns()
    # [ get_Nvl_fast(A, set(range(6)), set([])) for i in range(10**5)]
    stop = time.perf_counter_ns()
    print(f'Duration: {(stop-start)/1e9/10**5}')
