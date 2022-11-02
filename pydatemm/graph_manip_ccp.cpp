#pragma once
#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <Eigen/Core>
#include <Eigen/Dense>
#include "eigen_utils.h"




using Eigen::MatrixXd;
using namespace std;


set<int> get_nodes(const MatrixXd &X){
    // gets nodes in graph by looking at non-NaN indices
    vector<int> nodes;
    set<int> final_nodes;

    for (int i=0; i<X.rows();i++){
        for (int j=0; j<X.cols(); j++){
            if (!isnan(X(i,j))){
                final_nodes.insert(j);
                }
            }
        }
    return final_nodes;
    }

set<int> get_common_nodes(const MatrixXd &X, const MatrixXd &Y){
    set<int> X_nodes, Y_nodes, common_nodes;
    X_nodes = get_nodes(X);
    Y_nodes = get_nodes(Y);
    std::set_intersection(X_nodes.begin(), X_nodes.end(),
                          Y_nodes.begin(), Y_nodes.end(),
                           inserter(common_nodes, common_nodes.begin())
                           );
    return common_nodes;
    }


int check_for_one_common_edge(const set<int> &common_nodes, const MatrixXd &X, const MatrixXd &Y){
    /*
    common_nodes : set with only 2 elements in it. 
    */
    int i,j;
    int na, nb;
    bool one_edge_match, symmetric_match;
    vector<int> commonnodes;
    for (int a : common_nodes){
        commonnodes.push_back(a);
        }

    int num_common_edges = 0;
    for (int i=0; i<commonnodes.size(); i++){
        for (int j=i+1; j<commonnodes.size(); j++){
            na = commonnodes[i];
            nb = commonnodes[j];
            one_edge_match = X(na,nb) ==  Y(na,nb);
            symmetric_match = X(nb,na) == Y(nb,na);
            if (one_edge_match & symmetric_match){
                num_common_edges += 1;
                }            
            }
        }
    if (num_common_edges == 1){
        return 1;
        }else if (num_common_edges == 2){
        return 1;
        }
    else{
    return -1;
        }
        
}

int ccg_definer(const MatrixXd &X, const MatrixXd &Y){
    int relation;
    set<int> common_nodes;
    common_nodes = get_common_nodes(X,Y);
    if (common_nodes.size()>=2){
        if (common_nodes.size()<3){
            relation = check_for_one_common_edge(common_nodes,X,Y);
            }
        else{
            // all nodes the same
            relation = -1;
            }
        }
    else{
        // if # common nodes <2
        relation = -1;
        }
    return relation;
    }

MatrixXd make_ccg_matrix(const vector<MatrixXd> &all_cfls){
    int n_cfls = all_cfls.size();
    int relation=0;
    
    MatrixXd ccg_mat(n_cfls, n_cfls);
    for (int i=0; i<n_cfls; i++){
        for (int j=i+1; j<n_cfls; j++){
            relation = ccg_definer(all_cfls[i], all_cfls[j]);
            ccg_mat(i,j) = relation;
            ccg_mat(j,i) = relation;
            }
        }
    return ccg_mat;
    }

vector<vector<int>> mat2d_to_vector(MatrixXd X){
    cout << "Here we are.." << endl;
    int num_cfls = X.rows();
    vector<int> vec_data(num_cfls);
    vector<vector<int>> to_vect (num_cfls);
    for (int i=0; i<num_cfls; i++){
        for (int j=0; j<num_cfls; j++)
            {
            if (!isnan(X(i,j))){ 
                vec_data[j] = (int)X(i,j);
            }else{
                vec_data[j] = -1; // 
            }
            
            }
        to_vect[i] = vec_data;
        }
    cout << "Done we are.." << endl;
    return to_vect;
    }

MatrixXd combine_graphs(const set<int> &solution, const vector<MatrixXd> &all_cfls){
    int nchannels = all_cfls[0].rows();
    MatrixXd joint_graphs(nchannels, nchannels);
    vector<vector<int>> nonan_inds;
    for (auto graph_id : solution){
        nonan_inds = get_nonans(all_cfls[graph_id]);
        for (auto each :  nonan_inds){
                if (!each[0]==each[1]){
                joint_graphs(each[0], each[1]) = all_cfls[graph_id](each[0], each[1]);
                    }
            }                
        
        }
    return joint_graphs;
    }


struct tde_w_channels{
    vector<double> tde_data;
    vector<int> channels;
    };

tde_w_channels get_tde(MatrixXd array_geom, MatrixXd X, const double c=343.0){
    /* outputs a struct with 
     vector of tde_out 
     and channels
    @array_geom : xyz of mics
    @X : TDE graph
    @c : speed of sound in m/s. Defaults to 343 m/s.
    @output : struct with tde_data and channels attributes. 
    
    See Also
    @tde_w_channels 
    */
    vector<int> channels;
    tde_w_channels output;
    
    int num_rows = X.rows();
    output.channels.push_back(0);
    for (auto each : array_geom.row(0)){
        output.tde_data.push_back(each);
        }
    for (int j=0; j<num_rows; j++){
        if (!isnan(X(0,j)))
            {
            output.channels.push_back(j);
            // mic xyz of that channel
            for (auto each : array_geom.row(j)){
                output.tde_data.push_back(each);
                }
            }
        }
    
    // now append all the TDE data
    for (int j : output.channels){
        if (!j==0){
            output.tde_data.push_back(X(0,j)*c);
            }
        }
    return output;
    }


struct tde_data{
    map<int, vector<vector<double>> > tde_in;
    map<int, vector<set<int>> > cfl_ids;
    
    };

tde_data chunk_create_tde_data(vector<set<int>> comp_solns, MatrixXd array_geom,
                                                     vector<MatrixXd> all_cfls, double c=343.0){
    // outputs a struct with 2 maps formatted by tde-vector and cfl-ids.
    tde_data formatted_tde;
    int num_solns = comp_solns.size();
    int n_channels = all_cfls[0].rows();
    MatrixXd combined_tdemat(n_channels, n_channels);
    tde_w_channels tde_out;
    bool key_not_in_map;
    for (int i=0; i<num_solns; i++){
        combined_tdemat = combine_graphs(comp_solns[i], all_cfls);
        tde_out = get_tde(array_geom, combined_tdemat, c);
        n_channels = tde_out.channels.size();
        // if the n_channels key is not yet in the map
        key_not_in_map = formatted_tde.tde_in.find(n_channels)==formatted_tde.tde_in.end();
        if (key_not_in_map)
            {
            cout << "initialising: " << n_channels << "at index: " << i<<  endl;
            formatted_tde.tde_in[n_channels] = {};
            formatted_tde.tde_in.at(n_channels).push_back(tde_out.tde_data);
            formatted_tde.cfl_ids[n_channels] = {};
            formatted_tde.cfl_ids.at(n_channels).push_back(comp_solns[i]);
            }
        else{
            formatted_tde.tde_in.at(n_channels).push_back(tde_out.tde_data);
            formatted_tde.cfl_ids.at(n_channels).push_back(comp_solns[i]);
            }
        }

    return formatted_tde;
    }

/*
def chunk_create_tde_data(compatible_solutions, all_cfls, **kwargs):
    raw_tde_by_channelnum = {}
    cfl_ids = {} # for troubleshooting and error tracing
    for i, compat_cfl in enumerate(compatible_solutions):
        source_tde, channels = get_tde([all_cfls[j] for j in compat_cfl])
        d = source_tde[1:,0]*kwargs['vsound']
        numchannels = len(channels)
        tde_data = np.concatenate((kwargs['array_geom'][channels,:].flatten(), d))
        if raw_tde_by_channelnum.get(numchannels) is None:
            raw_tde_by_channelnum[numchannels] = []
            raw_tde_by_channelnum[numchannels].append(tde_data)
            cfl_ids[numchannels] = []
            cfl_ids[numchannels].append(compat_cfl)
        else:
            raw_tde_by_channelnum[numchannels].append(tde_data)
            cfl_ids[numchannels].append(compat_cfl)
    tde_by_channelnum = {}
    for nchannels, tde_data in raw_tde_by_channelnum.items():
        tde_by_channelnum[nchannels] = np.row_stack(tde_data)
    return tde_by_channelnum, cfl_ids
*/



/*int main(){
    MatrixXd x(4,4);
    MatrixXd y(4,4);
    x(1,0) = 1.5; x(0,1) = 1.5;
    
    x(1,2) = 3; x(2,1) = x(1,2);
    // 2 nodes
    x(2,0) = 1.5; x(0,2) = 1.5;
    y(2,0) = x(2,0); y(0,2) = y(2,0);
    
    y(2,3) = 1.5; y(3,2) = y(2,3);
    y(0,3) = 1.5; y(3,0) = y(0,3);
    
    set<int> xnodes = get_nodes(x);
    set<int> ynodes = get_nodes(y);
    
    std::cout << x << std::endl;
    
    std::cout << "x nodes " << std::endl;
    for (auto i : xnodes){
        std::cout << i << ", ";
        }
    
    std::cout << "\n y nodes " << std::endl;
    for (auto i : ynodes){
        std::cout << i << ", ";
        }
    set<int> commonnodes  = get_common_nodes(x,y);
    cout << " \n common nodes " << endl;
    for (auto k: commonnodes){
        cout << k << ", " ;
        }
    // check for common edges
    int common = check_for_one_common_edge(commonnodes, x, y);
    cout << "common edge check output " << common << endl;
    cout << "CCG definer output" << endl;
    
    int ccg_relation = ccg_definer(y,y);
    cout << "\n " << ccg_relation << endl;
    
    vector<MatrixXd> all_cfls(4);
    all_cfls[0] = x;
    all_cfls[1] = x;
    all_cfls[2] = y;
    all_cfls[3] = y;
     
    MatrixXd abc = make_ccg_matrix(all_cfls);
    cout << "abc \n \n" << abc;
    
    vector<vector<int>> mm = mat2d_to_vector(abc);
    
    cout << "\n " << endl;
    vector<vector<int>> ii = get_nonans(y);
    for (auto a : ii){
        cout << "\n " << endl;
        for (auto i : a){
            cout << i << ", ";
            }
        cout << endl;
        }
        
    set<int> combined = {0,2};
    MatrixXd uu = combine_graphs(combined, all_cfls);
    cout << uu << endl;
    
    return 0;
    }*/
