#pragma once
#include <vector>
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <stdexcept>

using Eigen::VectorXd;
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

int get_nmics(const vector<double> tde_data){
    
    if ((tde_data.size()+1) % 4 == 0){
		int nmics = (tde_data.size()+1)/4;
		return nmics;}
    else{
	std::cout << "Invalid TDE vector: " << tde_data.size() << " elements." << std::endl;
		throw std::invalid_argument( "Unable to calculate Nmics" );
		}
	}

VectorXd to_VXd(const vector<double> &Vd){
	VectorXd VXd(Vd.size());
	for (int i=0; i<Vd.size(); i++){
		VXd[i] = Vd[i];
	}
	return VXd;
	}

vector<double> to_vectdouble(const VectorXd &VXd)
	{
	vector<double> vectdouble(VXd.size());
	//VectorXd::Map(&vectdouble[0], v1.size()) = VXd
	for (int i=0; i < VXd.size(); i++)
	{ 
		vectdouble[i] = VXd[i];
	}
	return vectdouble;
	}