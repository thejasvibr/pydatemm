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

A = np.array([[ 0, 1, 0, 0,-1,-1],
              [ 1, 0, 1, 1, 0, 1],
              [ 0, 1, 0,-1, 1, 0],
              [ 0, 1,-1, 0,-1, 0],
              [-1, 0, 1,-1, 0, 1],
              [-1, 1, 0, 0, 1, 0]])

def get_Nvl(Acc, V, l):
    Nvl = []
    if len(l)>0:
        for v in V:
            for u in l:
                if Acc[v-1,u-1]==1:
                    Nvl.append(v)
                elif Acc[v-1,u-1]==-1:
                    if v in Nvl:
                        Nvl.pop(Nvl.index(v))
    else:
        Nvl = deepcopy(V)
    return set(Nvl)

def get_NOT_Nvl(Acc, V, l):
    N_not_vl = []
    if len(l)>0:
        for v in V:
            for u in l:
                if Acc[v-1,u-1]==-1:
                    N_not_vl.append(v)
                elif Acc[v-1,u-1]==1:
                    if v in N_not_vl:
                        N_not_vl.pop(N_not_vl.index(v))
    else:
        N_not_vl = []
    return set(N_not_vl)


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
    N_vl, N_not_vl = get_Nvl(Acc, V, l), get_NOT_Nvl(Acc, V, l)
    print(f'l:{l}, X:{X}, V:{V}, N_vl:{N_vl}, N_notvl:{N_not_vl}')
    solutions_l = []
    if len(N_vl) == 0:
        solutions_l.append(l)
        print(f'\n yes \n solution: {l}')
    else:
        # remove conflicting neighbour
        V = V.difference(N_not_vl)
        # unvisited compatible neighbours
        Nvl_wo_X = N_vl.difference(X)
        for n in Nvl_wo_X:
            Vx = V.difference(set([n]))
            lx = l.union(set([n]))
            solution = combine_all(Acc, Vx, lx, X)
            if solution:
                solutions_l.append(solution)
            X = X.union(set([n]))
    return solutions_l

#V = set(range(1,7)); Acc = deepcopy(A); V = set([]); l = set([]); X=set([])


qq = combine_all(A, set([1,2,3,4,5,6]), set([]), set([]))
# qq = combine_all(A, set([1,3,4,5,6]), set([2]), set([1]))
# qq = combine_all(A, set([1,2,3,4,6]), set([5]), set([1,2,3,4]))        

