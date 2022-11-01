#include <omp.h>
#include <iostream>
#include <chrono>
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "eigen_utils.h"
#include "tdoa_residual.h"

using Eigen::MatrixXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::Vector3d, Eigen::Vector2d;
using Eigen::VectorXd;
using Eigen::ArithmeticSequence;
using Eigen::seq;
using namespace std;

Vector2d  solve_eqn_22(const VectorXd &a, const VectorXd &b){
	// Inputs:
	// VectorXd a,b
	// nmics-1 length vector
	// Output:
	// Vector2d Rs12
	// Radius from 0th sensor.
	Vector2d Rs12;

	double term1, term2, bsquare_term, term2_ii;
	double term2_i, denominator, numerator1, numerator2;
	
	term1 = (a.array()*b.array()).sum();
	term2_i = pow(term1, 2);
	bsquare_term = b.array().pow(2).sum() - 1 ;
	term2_ii = bsquare_term*(a.array().pow(2).sum());
	term2 = sqrt(term2_i - term2_ii);

	denominator = bsquare_term;
	numerator1 = term1 + term2;
	numerator2 = term1 - term2;
	Rs12(0) = numerator1/denominator;
	Rs12(1) = numerator2/denominator;
	return Rs12;
}

VectorXd choose_correct_mpr_solutions(const VectorXd &mic_ntde_raw, const Vector2d &Rs_12,
									const VectorXd &a, const VectorXd &b){
	/*
	See Section C of Malanowski & Kulpa 2012.
	*/
	VectorXd xs(6);
	int num_Rs_position=0;
	int positive_ind;

	for (int i=0; i<2; i++){
		if (Rs_12(i)>0){
		num_Rs_position += 1;
		positive_ind = i;
		}
	}

	if (num_Rs_position==2){
		xs.head(3) << a - b*Rs_12(0) + mic_ntde_raw.head(3);
		xs.tail(3) << a - b*Rs_12(1) + mic_ntde_raw.head(3);
		
	}else if(num_Rs_position==1){
		xs.head(3) << a - b*Rs_12(positive_ind) + mic_ntde_raw.head(3);
	}else{
	// do nothing. 
	}

	return xs;
}


vector<double> mpr2003_optim(const vector<double> &mic_ntde_raw, const double &c=343.0){
    int nmics = get_nmics(mic_ntde_raw);
	vector<double> out;
	VectorXd mic_ntde_vx_raw = to_VXd(mic_ntde_raw); // without any subtraction
	VectorXd mic_ntde = to_VXd(mic_ntde_raw);
	MatrixXd S(nmics-1,3), invS_t_S(nmics-1,3);
	MatrixXd S_t(nmics-1,3), inv_StS_St(nmics-1,3);
	VectorXd di(nmics-1),z(nmics-1),a(nmics-1), b(nmics-1);
	Vector2d Rs_12;
	VectorXd xs(6);

    if (nmics <= 3){
        throw std::invalid_argument( "Less than or equal to 3 mics detected. Aborting." );
		}
    int position_inds = nmics*3;  
   	VectorXd mic0 = mic_ntde.head(3);
   	di = mic_ntde.tail(nmics-1);

   	MatrixXd R_inv(3, nmics-1);
   	ArithmeticSequence starts = seq(3, position_inds-3, 3);
   	ArithmeticSequence stops = seq(5, position_inds-1, 3);
   	for (int i=0; i<starts.size(); i++){
   		mic_ntde(seq(starts[i],stops[i])) +=  -mic0;
   		}
    // 0th sensor is assumed to be at origin
	S = mic_ntde(seq(3,position_inds-1)).reshaped(3,nmics-1).transpose();
	ArrayXXd S_sq_rowsum(S.rows(), S.cols());
	S_sq_rowsum = S.array().pow(2).rowwise().sum();
    // eqn. 12

    z = S_sq_rowsum - di.array().pow(2); // row-wise sum of S^2

	z *= 0.5;
	//eqn. 17 - without the weighting matrix R
	S_t = S.replicate(1,1).transpose();
	invS_t_S = (S_t*S).inverse();
	inv_StS_St = invS_t_S*S_t;
	a = inv_StS_St*z;
	// eqn. 22
	b = inv_StS_St*di;
	//eqn. 22
	Rs_12 = solve_eqn_22(a, b);
    // substitute Rs into eqn. 19
    xs = choose_correct_mpr_solutions(mic_ntde_vx_raw, Rs_12, a,b);
	//NEED TO CALCULATE TDOA RESIDUAL OUT HERE!!

	for (auto ii : xs){
		out.push_back(ii);
		}
	return out;
}


int main(){
	
	VectorXd mic_array(15);
	mic_array << 0,0,1.0,
						  0,1.0,0,
						  1.0,0,0,
						  1.0,1.2,0,
						  0.7383041911100019,
						  0.8769856518476695,
						0.6424417440905668;
	vector<double> mvect;
	
	for (auto i : mic_array){
		mvect.push_back(i);
	}
	//cout << "mic array \n " << mic_array.transpose() << endl;
	MatrixXd qq(3,4);
	qq = mic_array(seq(0,11)).reshaped(3,4).transpose();
	vector<double> oo;
	oo = mpr2003_optim(mvect);
	for (auto ii : oo){
		cout << ii << ", " ;
	}
	cout << "" << endl;
		
	
	
}
