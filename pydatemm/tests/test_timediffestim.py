#!/usr/bin/env python

"""
Tests for the time difference estimation module
"""

from itertools import combinations
import unittest

import numpy as np
import pydatemm.timediffestim as tde
import pydatemm.simdata as simdata
import scipy.spatial as spatial

class TestGeomValidTDES(unittest.TestCase):

    def setUp(self):
        """
        make simulated array and TDOAs with valid
        and invalid entries
        """
        # make array
        self.tristar = simdata.make_tristar()
        self.tdes = {}
        channel_pairs = list(combinations(range(4),2))
        # generate distance and delay matrix
        self.distmat = spatial.distance_matrix(self.tristar, self.tristar)
        self.vsound = 340
        self.delaymat = self.distmat/self.vsound
        # assign 3 'valid' TDES to each channel pair
        for pair in channel_pairs:
            ch1,ch2 = pair
            maxdelay = self.delaymat[ch1,ch2]
            possible_valid_tdes = np.linspace(-maxdelay,maxdelay,20)
            self.tdes[pair] = np.random.choice(possible_valid_tdes, 3).tolist()
            # add spurious TDEs
            for i in range(2):
                invlid_tdes = np.linspace(-(maxdelay+1), maxdelay+1, 20)
                self.tdes[pair].append(np.random.choice(invlid_tdes,3))

    def test_tristar_case(self):
        self.check_tdes_are_valid()

    def test_empty_tde_case(self):
        '''When there is no tdoa in one of the channel pairs  '''
        self.tdes[(0,1)] = []
        self.check_tdes_are_valid()
    
    def check_tdes_are_valid(self):
        tdes_geomvalid = tde.geometrically_valid(self.tdes,
                                array_geom=self.tristar,
                                v_sound=self.vsound)
        leq_maxdelay = []
        for ch_pair, tdes in tdes_geomvalid.items():
            ch1, ch2 = ch_pair
            maxdelay = self.delaymat[ch1,ch2]
            all_leq = np.all(abs(np.array(tdes))<=maxdelay)
            leq_maxdelay.append(all_leq)
        self.assertTrue(np.all(leq_maxdelay))

if __name__ == '__main__':
    unittest.main()