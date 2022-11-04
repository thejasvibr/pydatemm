#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
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

struct output_lists{
    // Provides simplified '1D' representation of the output data 
    vector<vector<double>> sources;
    vector<vector<double>> tde_in;
    vector<set<int>> cfl_ids;
    };



 

output_lists localise_sounds_v3(const int &num_cores, const MatrixXd &array_geom,
                         const vector<set<int>> &compatible_solutions, const vector<MatrixXd> &all_cfls,
                                  double c=343.0){
    /*Formats CCG solutions and generates candidate locations. 
    
    @param num_cores Number of cores to use when running parallelised code
    @param array_geom (Nmics,3) matrix with xyz coordinates of microphones
    @param compatible_solutions CCG solutions shown as compatible cFL indices
    @param all_cfls All found cFLs
    @param c Speed of sound in m/s. Defaults to 343.0 m/s
    @return output_lists Struct with three different maps. 
    
    See Also: output_lists
    
    */
    output_lists final_output;
    tde_data processed_tde;
    vector<vector<double>> temp_sources;
    vector<double> temp_4ch_sources;
    vector<double> temp_one_source(4);
    
    processed_tde = chunk_create_tde_data(compatible_solutions, array_geom, all_cfls);
    
    for (auto x : processed_tde.tde_in){
        if (x.first>4){
            // here the size of tde inputs matches the size of 
            // other data exactly
            temp_sources  = pll_sw_optim(x.second, num_cores, c);

            for (int i=0; i<x.second.size(); i++){
                final_output.sources.push_back(temp_sources[i]);
                final_output.tde_in.push_back(x.second[i]);
                final_output.cfl_ids.push_back(processed_tde.cfl_ids[x.first][i]);
                }
            }
        else if(x.first==4){
            for (int i=0; i<x.second.size(); i++){
                // with 4 channels there's always upto 2 valid solutions.
                temp_4ch_sources = mpr2003_optim(x.second[i], c);
                // first source
                copy(temp_4ch_sources.begin(), temp_4ch_sources.begin()+4, temp_one_source.begin());
                final_output.sources.push_back(temp_one_source);
                final_output.tde_in.push_back(x.second[i]);
                final_output.cfl_ids.push_back(processed_tde.cfl_ids[x.first][i]);
                
                // second source
                copy(temp_4ch_sources.begin()+4, temp_4ch_sources.end(), temp_one_source.begin());
                final_output.sources.push_back(temp_one_source);
                final_output.tde_in.push_back(x.second[i]);
                final_output.cfl_ids.push_back(processed_tde.cfl_ids[x.first][i]);
                }
            }

        }
       return final_output;
   }





summary_data localise_sounds_v2(const int &num_cores, const MatrixXd &array_geom,
                         const vector<set<int>> &compatible_solutions, const vector<MatrixXd> &all_cfls,
                                  double c=343.0){
    /*Formats CCG solutions and generates candidate locations. 
    
    @param num_cores Number of cores to use when running parallelised code
    @param array_geom (Nmics,3) matrix with xyz coordinates of microphones
    @param compatible_solutions CCG solutions shown as compatible cFL indices
    @param all_cfls All found cFLs
    @param c Speed of sound in m/s. Defaults to 343.0 m/s
    @return final_output Struct with three different maps. 
    
    See Also: summary_data
    
    */
    summary_data final_output;
    tde_data processed_tde;
    processed_tde = chunk_create_tde_data(compatible_solutions, array_geom, all_cfls);

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