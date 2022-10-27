#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;


set<int> get_nodes(const MatrixXd &X){
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
    
    //i = commonnodes[0];
    //j = commonnodes[1];
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
    return 0;
        }
        
    /*if(common_nodes.size()==2){
        bool one_edge_match = X(i,j)==Y(i,j);
        bool symmetric_match = X(j,i)==Y(j,i);
        if (one_edge_match & symmetric_match){
            return 1;
            }else{
            return -1;
            }
    }
    else{
        std::cout << "Number of nodes != 2 \n"<< endl;
        for (int a:common_nodes){
            cout << a << ", " << endl;
            }
        throw std::invalid_argument( " Num. nodes !=2 - unable to check for 1 common edge" );
    
        }*/
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
            vec_data[j] = (int)X(i,j);
            }
        to_vect[i] = vec_data;
        }
    return to_vect;
    }



int main(){
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
    return 0;
    }
