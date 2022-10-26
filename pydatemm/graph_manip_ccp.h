#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;


set<int> get_node(const MatrixXd &X){
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

set<int> get_common_node(const MatrixXd &X, const MatrixXd &Y){
    set<int> X_nodes, Y_nodes, common_nodes;
    X_nodes = get_node(X);
    Y_nodes = get_node(Y);
    std::set_intersection(X_nodes.begin(), X_nodes.end(),
                          Y_nodes.begin(), Y_nodes.end(),
                           inserter(common_nodes, common_nodes.begin())
                           );
    return common_nodes;
    }

int check_for_one_common_edge(const MatrixXd X, const MatrixXd Y){
    
    
    

    }


int ccg_definer(const MatrixXd &X, const MatrixXd &Y){
    int relation;
    vector<int> common_nodes;
    common_nodes = get_common_nodes(X,Y);
    if (common_nodes.size()>=2){
        if (common_nodes.size()<3){
            relation = check_for_one_common_edge(X,Y);
            }
        else{
            // all nodes the same
            relation = -1;
            }
        }
    else{
        relation = -1;
        }
    return relation;
    }

