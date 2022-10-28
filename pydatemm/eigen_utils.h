#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;

vector<vector<int>> get_nonans(const MatrixXd &X){
    // Gets the indices of the non NaN entries. 
    vector<vector<int>> all_inds;
    vector<int> row_inds;
    for (int i=0; i<X.rows(); i++){
        
        for (int j=0; j<X.rows(); j++){
            if (!isnan(X(i,j))){
                row_inds = {i,j};
                all_inds.push_back(row_inds);
            }
        }
    }
    return all_inds;
    }

//vector<int> get_nonans1d()