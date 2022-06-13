# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:23:18 2022

@author: theja
"""
from itertools import product
from copy import deepcopy
import networkx as nx
from pydatemm.timediffestim import *
from pydatemm.raster_matching import multichannel_raster_matcher
from pydatemm.triple_generation import mirror_Pprime_kl, generate_consistent_triples
from pydatemm.triple_generation import choose_consistent_triples
from pydatemm.graph_synthesis import sort_triples_by_quality
from pydatemm.tdoa_quality import residual_tdoa_error as ncap
from pydatemm.simdata import make_chirp
import pyroomacoustics as pra
from scipy.spatial import distance_matrix, distance
euc_dist = distance.euclidean
import pandas as pd
import soundfile as sf
import tqdm 
import time
#%%

def printout_edge_weights(graph_list):
    all_weights= []
    for G in graph_list:
        weights = []
        for u, v, w in G.edges(data=True):
            weights.append(w['tde']*1e3)
        all_weights.append(weights)
    return all_weights
#%%
from itertools import permutations
print('starting sim audio...')
seednum = 78464 # 8221, 82319, 78464
np.random.seed(seednum) # what works np.random.seed(82310)
array_geom = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
array_geom = array_geom[:7,:]

nchannels = array_geom.shape[0]
fs = 192000

paper_twtm = 1e-5
kwargs = {'twrm': paper_twrm,
          'twtm': paper_twtm,
          'nchannels':nchannels,
          'fs':fs}
room_dim = [4,2,2]

# We invert Sabine's formula to obtain the parameters for the ISM simulator

rt60 = 0.1
e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
room = pra.ShoeBox(room_dim, fs=kwargs['fs'],
                   materials=pra.Material(0.5), max_order=1)
#mic_locs = np.random.normal(0,2,3*kwargs['nchannels']).reshape(3,nchannels)
# array_geom = np.abs(np.random.normal(0,1,3*nchannels).reshape(3,nchannels))

kwargs['array_geom'] = array_geom
room.add_microphone_array(array_geom.T)

# add one source
pbk_signals = [make_chirp(chirp_durn=0.05, start_freq=80000, end_freq=50000)*0.5,
               make_chirp(chirp_durn=0.05, start_freq=40000, end_freq=10000)*0.5]
source1 = [1.67,1.66,0.71]
source2 =  [1.67,1.66,0.71] # [2.72,0.65,1.25]#
sources = np.vstack((source1, source2))
source_positions = [source1, source2]
for i,each in enumerate(source_positions):
    room.add_source(position=each, signal=pbk_signals[i], delay=i*0.02)
room.compute_rir()
room.simulate()
audio = room.mic_array.signals.T
multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
multich_ac = generate_multich_autocorr(audio)
cc_peaks = get_multich_tdoas(multich_cc, min_height=10, fs=fs, min_peak_diff=1e-5)

def sort_cc_peaks_by_quality(multich_ccpks, top_N=15):
    sorted_cc_peaks = {}
    for channel_pair, peaks in multich_ccpks.items():
        index_sort = np.argsort([each[-1] for each in peaks])[::-1]
        sorted_cc_peaks[channel_pair] = []
        for i,index in enumerate(index_sort):
            sorted_cc_peaks[channel_pair].append(peaks[index])
        sorted_cc_peaks[channel_pair] = sorted_cc_peaks[channel_pair][:top_N]
    return sorted_cc_peaks

cc_peaks_sorted = sort_cc_peaks_by_quality(cc_peaks, 20)


multiaa = get_multich_aa_tdes(multich_ac, fs=fs,
                              min_height=3) 

def fake_raster_matcher(multich_Pkl, mm, **kwargs):
    # keep only geometrically valid TDOAs across channels
    geomvalid_Pkl = geometrically_valid(multich_Pkl, **kwargs)

    multich_Pprime_kl = {}
    for ch_pair, Pkl in geomvalid_Pkl.items():
        Pprime_kl = []
        for (a,b,c) in Pkl:
            new = (a,b,c,c)
            Pprime_kl.append(new)
        multich_Pprime_kl[ch_pair] = Pprime_kl
    return multich_Pprime_kl

print('raster matching...')

tdoas_rm = fake_raster_matcher(cc_peaks, multiaa,
                                       **kwargs)
tdoas_mirrored = mirror_Pprime_kl(tdoas_rm)    
print('triple generation')
#%%
mic2source = distance_matrix(np.vstack((source1, array_geom)), np.vstack((source1, array_geom)))[1:,0]
mic2source

deltaR = np.zeros((nchannels, nchannels))
for i,j in product(range(nchannels), range(nchannels)):
    deltaR[i,j] = mic2source[i]-mic2source[j]
full_graph = deltaR/0.340

sourceG = nx.from_numpy_array(full_graph)

def get_triple_weights(triple, tde_array):
    '''tde_array: NchannelxNchannel array with tdes'''
    a,b,c = triple
    return (tde_array[a,b], tde_array[b,c], tde_array[c,a])

def plot_graph_w_labels(graph, curr_ax):
    pos = nx.circular_layout(graph)
    nx.draw_circular(graph, with_labels=True)
    weight_labels = {}
    for e in graph.edges():
        try:
            weight_labels[e] = np.around(graph.edges[e]['tde']*1e3, 3)
        except KeyError:
            weight_labels[e] = np.around(graph.edges[e]['weight'], 3)
    nx.draw_networkx_edge_labels(graph, pos,edge_labels=weight_labels,
                                 ax=curr_ax)

plt.figure()
a0 = plt.subplot(111)
plot_graph_w_labels(sourceG, a0)

#%%
G = nx.complete_graph(nchannels)
minspan_G = nx.minimum_spanning_tree(G)
main_node = [0]
co_tree = nx.complement(minspan_G)
# plt.figure()
# plt.subplot(211)
# nx.draw_circular(minspan_G, with_labels=True)
# plt.subplot(212)
# nx.draw_circular(co_tree, with_labels=True)
#%%
# make all the Fundamental Loops by adding one edge from the co-tree
fundamental_loops = []
for edge in co_tree.edges():
    fl_nodes = tuple(set(main_node + list(edge)))
    fundamental_loops.append(fl_nodes)
#%%
# Assemble all possible consistent triples from the fundamental loops!
kwargs = {'twtm':1e-5}
cFLs = {}
for each in fundamental_loops:
    funda_loops = choose_consistent_triples(each, tdoas_mirrored, **kwargs)
    cFLs[tuple(each)] = []
    for every in funda_loops:
        cFLs[tuple(each)].append(every[1:])
print([(each, len(vals))for each, vals in cFLs.items()])
# #%% And now filter all consistent triples which are fundamental loops
# cFLs = [each for each in sorted_triples_full if set(each.nodes()) in fundamental_loops]
#%%
# According to Kreissig & Yang 2012 (ICASSP) let's begin sequentially merging
# cFLs together to get larger consistent graphs. 
# e.g. let's say there are 4 cFLs (A,B,C,D), first get all possible
# A-B merges. Then all possible A-B-C merges, and then all possible A-B-C-D
# merges. 
from pydatemm.graph_synthesis import two_common_nodes_and_edges
from pydatemm.triple_generation import make_channel_pairs_from_triple

def four_common_edges_two_nodes(X,Y):
    xx = nx.intersection(X,Y)
    return np.all([len(xx.edges)==4, len(xx.nodes)==2])

def two_common_nodes_one_edge(X,Y):
    xx = nx.intersection(X,Y)
    return np.all([len(xx.edges)==1, len(xx.nodes)==2])

def make_triple_graph_from_channel_pairs(triple_name, edge_info):
    channel_pairs = make_channel_pairs_from_triple(triple_name)
    triple_graph = nx.DiGraph()
    for pair, tde in zip(channel_pairs, edge_info):
        triple_graph.add_edge(pair[0], pair[1],
                              **{'tde': tde[0], 'peak_score': tde[1]})
        # triple_graph.add_edge(pair[1], pair[0],
        #                       **{'tde': -tde[0], 'peak_score': -tde[1]})
    return triple_graph

cfl_triples = {}
for triple, triple_instances in cFLs.items():
    cfl_triples[triple] = []
    for each in triple_instances:
        triple_graph = make_triple_graph_from_channel_pairs(triple, each)
        cfl_triples[triple].append(triple_graph)

#%%
# Let's focus on 015, and 016 which should be mergeable. We'll search 
# to see if the 'source' triples taken from the full_graph exist in the first place.
trip_015 = np.array(get_triple_weights((0,1,5), full_graph))
trip_016 = np.array(get_triple_weights((0,1,6), full_graph))

def best_trip_match(target_trip_weights, triples):
    w_trips = printout_edge_weights(triples)
    target_distance = [euc_dist(each, target_trip_weights) for each in w_trips]
    best_fit_ind = np.argmin(target_distance)
    return best_fit_ind, triples[best_fit_ind]

ind_015, trip_015 = best_trip_match(trip_015, cfl_triples[(0,1,5)])
ind_016, trip_016 = best_trip_match(trip_016, cfl_triples[(0,1,6)])
#%%
all_fl_pairs = list(combinations(fundamental_loops,2))
all_possible = []
for (fl_a, fl_b) in all_fl_pairs:
    one_two_combis = list(product(cfl_triples[fl_a][:50], cfl_triples[fl_b][:50]))
    comp_cfls = []
    for i, (each, every) in enumerate(one_two_combis):
        if two_common_nodes_one_edge(each, every):
            #print(i)
            comp_cfls.append([each, every])
    all_possible.append(comp_cfls)
#%%
# Alternate merging strategy as in Fig. 2 of Kreissig & yang 2012
three_fls_joint = []
for each in tqdm.tqdm(cfl_triples[fundamental_loops[0]][:100]):
    for every in cfl_triples[fundamental_loops[1]][:100]:
        if two_common_nodes_one_edge(each, every):
            abcd = nx.compose(each, every)
            for tt in cfl_triples[fundamental_loops[2]][:100]:
                if two_common_nodes_one_edge(abcd, tt):
                    print('yay')
                    abcdef = nx.compose(abcd, tt)
                    three_fls_joint.append(abcdef)

#%%
# Need a way to run compatibilty checks 
def combine_graphs(X, Y):
    return nx.compose(X, Y)

def ccg_determine(X,Y):
    if two_common_nodes_one_edge(X, Y):
        return 1
    xx = nx.intersection(X,Y)
    if np.logical_and(len(xx.nodes)==2, len(xx.edges)==0):
        return -1
    elif len(xx.nodes)<2:
        return 0
    else:
        raise NotImplementedError('This case not implemented')
#%%
        

all_possible_trip_combis = product(*[cfl_triples[each][:25] for each in fundamental_loops[:4]])

for combiset in all_possible_trip_combis:
    tripb_ind = 1 
    tripa, tripb = combiset[0], combiset[1]
    while True:
        if two_common_nodes_one_edge(tripa, tripb):
            tripa = combine_graphs(tripa, tripb)
            tripb_ind += 1
            tripb = combiset[tripb_ind]
        else:
            break
            
#%%
# Let's build the compatibility-conflict graph for all the fundamental loops
fl_pairs = product(range(len(fundamental_loops)), range(len(fundamental_loops)))
ccg = np.zeros([len(fundamental_loops)]*2)

def determine_compatibility(X,Y):
    pass

for (i,j) in fl_pairs:
    X, Y = fundamental_loops[i], fundamental_loops[j]
    ccg[i,j] = ccg_determine(X,Y)

        
        
        

    
    




#%%
plt.figure(); a1 = plt.subplot(111)
plot_graph_w_labels(three_fls_joint[-1], a1)