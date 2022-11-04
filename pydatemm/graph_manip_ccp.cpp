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
using Eigen::all;
using namespace std;


set<int> get_nodes(const MatrixXd &X){
    // gets nodes in graph by looking at non-NaN indices
    set<int> final_nodes;
    
    for (int i=0; i<X.rows();i++){
        for (int j=0; j<X.cols(); j++){
            if (!isnan(X(i,j))){
                final_nodes.insert(j);
                final_nodes.insert(i);
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
                joint_graphs(each[1], each[0]) = joint_graphs(each[0], each[1]);
                    }
            }                
        
        }
    return joint_graphs;
    }


struct tde_w_channels{
    vector<double> tde_data;
    set<int> channels;
    };

tde_w_channels get_tde(const MatrixXd &array_geom, const MatrixXd &X, const double c=343.0){
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
    output.channels = get_nodes(X);
    
    for (auto i : output.channels){
        for (auto each : array_geom.row(i)){
            output.tde_data.push_back(each);
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

tde_data chunk_create_tde_data(const vector<set<int>> &comp_solns, const MatrixXd &array_geom,
                                                     const vector<MatrixXd> &all_cfls, double c=343.0){
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
            //cout << "initialising: " << n_channels << "at index: " << i<<  endl;
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
