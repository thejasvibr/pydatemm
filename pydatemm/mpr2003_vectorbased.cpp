#include <omp.h>
#include <iostream>
#include <chrono>
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
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::ArithmeticSequence;
using Eigen::seq;
using namespace std;


vector<double> mpr2003_optim(const vector<double> &mic_ntde_raw, const double &c=343.0){
    int nmics = get_nmics(mic_ntde_raw);
	cout << "num mics is " << nmics << endl;
	VectorXd mic_ntde_vx_raw = to_VXd(mic_ntde_raw); // without any subtraction
	VectorXd mic_ntde = to_VXd(mic_ntde_raw);
	MatrixXd S(nmics-1,3), S_t_S(nmics-1,3);
	MatrixXd S_t(nmics-1,3);
	ArrayXd di(nmics-1);
	ArrayXd z(nmics-1);
    if (nmics <= 3){
        throw std::invalid_argument( "Less than or equal to 3 mics detected. Aborting." );
		}
    int position_inds = nmics*3;  
   	VectorXd mic0 = mic_ntde.head(3);
   	di = mic_ntde.tail(nmics-1).array();

   	MatrixXd R_inv(3, nmics-1);
   	ArithmeticSequence starts = seq(3, position_inds-3, 3);
   	ArithmeticSequence stops = seq(5, position_inds-1, 3);
   	for (int i=0; i<starts.size(); i++){
   		mic_ntde(seq(starts[i],stops[i])) +=  -mic0;
   		}
	
	cout << "di " << di << endl;
	
    // 0th sensor is assumed to be at origin
	S = mic_ntde(seq(3,position_inds-1)).reshaped(3,nmics-1).transpose();
	ArrayXXd S_sq_rowsum(S.rows(), S.cols());
	S_sq_rowsum = S.array().pow(2).rowwise().sum();
	cout << "Ssq \n" << endl;
	cout << S_sq_rowsum << endl;
    // eqn. 12
	//cout<< pow(S,2) <<endl;
	
	//cout << "Ssq" << S_sq_rowsum << endl;
    z = S_sq_rowsum - di.pow(2); // row-wise sum of S^2
    cout << "z" << z << endl;
	z *= 0.5;
	cout << " z 0.5 \n " << z << endl;
	//inv_StS
	cout << "S \n" << S << endl;
	S_t = S.replicate(1,1).transpose();
	cout << "S_t \n" << S_t << endl;
	S_t_S.noalias() = S_t*S;
	cout << "S_t_S inver" << S_t_S.inverse() << endl;
    /*# eqn. 17 - without the weighting matrix R
    inv_StS = np.linalg.inv(np.dot(S.T,S));
    inv_StS_St = np.dot(inv_StS, S.T);
    a = np.dot(inv_StS_St, z);
    # eqn. 18
    b = np.dot(inv_StS_St, di);
    # eqn. 22
    Rs_12= solve_eqn_22(a, b);
    # substitute Rs into eqn. 19
    xs = choose_correct_mpr_solutions(mic_array, Rs_12, (a,b), di);
    return xs;*/
	vector<double> out = {0,1.5,2.4};
	return out;
}

int main(){
	
	VectorXd mic_array(15);
	mic_array << 0,0,1.0,
						  0,1.0,0,
						  1.0,0,0,
						  1.0,1.2,0,
						  -0.5684942243080222,
						  -0.1299449391377756,
						-1.1644047618815492;
	//mic_array(12) = -0.5684942243080222;
	//mic_array(13) = -0.1299449391377756;
	//mic_array(14) = -1.1644047618815492;

	vector<double> mvect;
	
	
	for (auto i : mic_array){
		mvect.push_back(i);
	}
	cout << "mic array \n " << mic_array.transpose() << endl;
	MatrixXd qq(3,4);
	qq = mic_array(seq(0,11)).reshaped(3,4).transpose();
	vector<double> oo;
	oo = mpr2003_optim(mvect);
	
	
}
