'''
Tests for graph synthesis
'''
import unitest
from pydatemm.graph_synthesis import *

#%% This example triplet-> quadruplet merger is from Fig 11 of Scheuing & Yang 2008

klm = [(0,1,2), (5,00), (-1, 00), (-4,00)]
klo = [(0,1,3), (5,00), (-7,00), (2, 00)]
kmo = [(0,2,3), (4,00), (-6,00), (2,00)]

m0 = make_triplet_graph(klm, 4)
m1 = make_triplet_graph(klo, 4)
m2 = make_triplet_graph(kmo, 4)
sum_trips = add_three_triplets(m0,m1,m2)

#%% Test cases for build full toads 
# 4, 5, 20 channels
