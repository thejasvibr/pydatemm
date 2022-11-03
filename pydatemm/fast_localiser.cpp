#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <Eigen/Core>
#include <Eigen/Dense>
#include "graph_manip_ccp.cpp"
#include "sw2002_vectorbased.h"
#include "mpr2003_vectorbased.h"
using namespace std;

struct summary_data{
    map<int, vector<vector<double>> > sources;
    map<int, vector<vector<double>> > tde_in;
    map<int, vector<set<int>> > cfl_ids;
    };

summary_data localise_sounds_v2(int num_cores, MatrixXd array_geom,
                         vector<set<int>> compatible_solutions, vector<MatrixXd> all_cfls,
                                  double c=343.0){
    summary_data final_output;
    tde_data processed_tde;
    processed_tde = chunk_create_tde_data(compatible_solutions, array_geom, all_cfls);
    vector<vector<double>> sources;
    vector<vector<double>> inputdata;
    cout << "what?" << endl;
    vector<double> onesoln;
    
    for (auto x : processed_tde.tde_in){
        if (x.first>4){
            final_output.sources[x.first] = pll_sw_optim(x.second, num_cores, c);
            }
        else if(x.first==4){
            final_output.sources[x.first] = many_mpr2003_optim(x.second, c);
            }

        }
    final_output.tde_in.insert(processed_tde.tde_in.begin(), processed_tde.tde_in.end()); 
    final_output.cfl_ids.insert(processed_tde.cfl_ids.begin(), processed_tde.cfl_ids.end()); 
       return final_output;
       }