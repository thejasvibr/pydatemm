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

summary_data localise_sounds_v21(int num_cores, MatrixXd array_geom,
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
            cout << "Nchannels: "<< x.first << endl;
            //final_output.sources[x.first] = pll_sw_optim(x.second, num_cores, c);
            inputdata = x.second;
            cout << "x.second[0].size: " << x.second[0].size() << endl;
            //onesoln = sw_matrix_optim(x.second[0], c);
            cout << "x.second[0] entries: " << endl;
            for (auto ii:x.second[0]){
                cout << ii << ", ";
                }
            cout << endl;
            
            
            //pll_sw_optim(inputdata, num_cores, c);
            }
        else if(x.first==4){
            // 
            cout << "bow!!" << endl;
            //final_output.sources[x.first] = many_mpr2003_optim(x.second, c);
            }

        }
    final_output.tde_in.insert(processed_tde.tde_in.begin(), processed_tde.tde_in.end()); 
    final_output.cfl_ids.insert(processed_tde.cfl_ids.begin(), processed_tde.cfl_ids.end()); 
    //final_output.tde_in = processed_tde.tde_in;
    //final_output.tde_in.merge(processed_tde.tde_in);
    //final_output.cfl_ids.merge( processed_tde.cfl_ids);
       return final_output;
       }
    
    
    
    /*all_sources = []
    all_cfls = []
    all_tdedata = []
    for (nchannels, tde_input) in tde_data.items():
        print('In For Loop', nchannels, tde_input.shape)
        if nchannels > 4:
            calc_sources = lo.pll_cppyy_sw2002(tde_input, num_cores, kwargs['vsound'])
            all_sources.append(calc_sources)
            all_cfls.append(cfl_ids[nchannels])
            all_tdedata.append(tde_input.tolist())
        elif nchannels == 4:
            fourchannel_cflids= []
            fourchannel_tdedata = []
            for i in range(tde_input.shape[0]):
                calc_sources = lo.row_based_mpr2003(tde_input[i,:])
                nrows = get_numrows(calc_sources)
                if nrows == 2:
                    all_sources.append(calc_sources[0,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                    all_sources.append(calc_sources[1,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                elif nrows == 1:
                    all_sources.append(calc_sources)
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                elif nrows == 0:
                    pass                
            all_cfls.append(fourchannel_cflids)
            all_tdedata.append(fourchannel_tdedata)
        else:
            pass # if <4 channels encountered
    if len(all_sources)>0:
        return np.row_stack(all_sources), list(chain(*all_cfls)), list(chain(*all_tdedata))
    else:
        return np.array([]), [], []*/
        
 
