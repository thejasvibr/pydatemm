'''
Tests for Raster Matching
=========================


Created on Wed May 25 08:23:51 2022
@author: thejasvi beleyur
'''
import unittest 
from pydatemm.raster_matching import * 




class TestMakePprime(unittest.TestCase):
    def setUp(self):
        self.paa = [(10, 10, 1),
                    (20, 20, 1),
                    (30, 30, 1)]
        self.pkl = [( 0,  0, 1),
                    (10, 10, 1),
                    (42, 42, 1)]
        self.twrm = 6
    def expected_pprime(self):
        pprime = [
        [(10, 10, 1), ((0, 0, 1),  (10, 10, 1))],
        [(30, 30, 1), ((10, 10, 1), (42, 42, 1))]
        ]
        return pprime
    def test_simple(self):
        pprime = make_Pprime(self.paa, self.pkl, twrm=self.twrm)
        self.assertEqual(pprime, self.expected_pprime())

def make_scheuingyang08_data():
    '''
    Generates the data behind the example for Fig. 18 in the Scheuing & Yang
    2008 paper
    '''
    p12 = [-81, -31, -4, 21, 48, 109, 162, 188, 267, 327,
                                  337, 347, 358, 438, 448]
    # all right pointing arrows
    p11 = [10, 21, 28, 142];
    # all left pointing arrows
    p22 = [35, 52, 79, 224]
    return p12, p11, p22

def format_p12_p11_p22_into_dicts():
    p12, p11, p22 = make_scheuingyang08_data()
    # for each tde, now make them into a peak object
    Pkl = [(each, each, 5 ) for each in p12]
    Pkk = [(each, each, 5 ) for each in p11]
    Pll = [(each, each, 5 ) for each in p22]
    return Pkl, Pkk, Pll

def expected_P12_prime():
    # eqn. 29
    return [-31, 21, 48, 267, 327, 337, 438]

class TestScheuingYang2008_rastermatching(unittest.TestCase):
    '''Checks if the example in Fig. 18 of the paper is replicated
    '''
    def setUp(self):
        self.pkl, self.pkk, self.pll = format_p12_p11_p22_into_dicts()
        self.expected_pp12 = expected_P12_prime()
        self.keyword_args = {'twrm':7}
    def test_rastermatching(self):
        output_peaks = channel_pair_raster_matcher(self.pkl, self.pkk,
                                                   self.pll,
                                                   **self.keyword_args)
        output_delays = list(map(lambda X: X[0], output_peaks))
        expected_peaks = expected_P12_prime()
        self.assertEqual(expected_peaks, output_delays)

if __name__ == '__main__':
    unittest.main()

    fs = 96000
    pkl, pkk, pll = format_p12_p11_p22_into_dicts()
    
    channel_pair_raster_matcher(pkl, pkk, pll, twrm=7/fs)