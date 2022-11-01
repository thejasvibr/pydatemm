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
	/*
	@param a, b Intermediate nmics-1 length Vectors
	@return Rs12 radii from potential sources to 0th sensor.
	*/
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
	@param mic_ntde_raw 1D vector containing the flattened 3d coordinates of the mic array
		along with the range differences wrt mic 0.
	@param Rs_12 Vector with 2 radial distances of potential sources to mic 0. 
	@param a,b nmics-1 intermediate vectors. 
	@return xs Vector with upto 6 valid values (when two potential solutions exist), 
	else all nans or only 3 valid values. 

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
	/*
	@param mic_ntde_raw Vector with nmics*3 + nmics-1 entries. The first nmics*3 entries are 
	the flattened coordinates of the microphones. The last nmics-1 are the range differences
	in metres wrt the 0th sensor. 
	@param c Optional. Speed of sound in m/s. Defaults to 343 m/s. 
	@return out Vector with upto 8 valid entries when both solutions exist (4 entries for each solution).
	4 entries correspond to x, y, z and TDOA-residual as defined in Scheuing & Yang 2008. 
		
	*/
    int nmics = get_nmics(mic_ntde_raw);
	vector<double> out;
	VectorXd mic_ntde_vx_raw = to_VXd(mic_ntde_raw); // without any subtraction
	VectorXd mic_ntde = to_VXd(mic_ntde_raw);
	MatrixXd S(nmics-1,3), invS_t_S(nmics-1,3);
	MatrixXd S_t(nmics-1,3), inv_StS_St(nmics-1,3);
	MatrixXd arraygeom(nmics,3);
	VectorXd di(nmics-1),z(nmics-1),a(nmics-1), b(nmics-1);
	Vector2d Rs_12;
	VectorXd xs(6);
	VectorXd xs_res(8);
	double residual_1, residual_2; // tdoa residual

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
	xs_res.head(3) = xs.head(3);
	xs_res(seq(4,6)) = xs.tail(3);
	arraygeom = mic_ntde_vx_raw(seq(0,3*nmics - 1)).reshaped(3,nmics).transpose();
	if (!isnan(xs_res(0))){
		xs_res(3) = residual_tdoa_error(mic_ntde_vx_raw.tail(nmics-1), xs.head(3), arraygeom, c);
	}
	if (!isnan(xs_res(4))){
		xs_res(7) = residual_tdoa_error(mic_ntde_vx_raw.tail(nmics-1), xs.head(3), arraygeom, c);
	}
	out = to_vectdouble(xs_res);
	return out;
}

vector<vector<double>> many_mpr2003_optim(const vector<vector<double>> &all_inputs, double c=343.0){
	/*
	The multi-input version of mpr2003_optim.
	@param all_inputs Vector of vectors containing various mic-array xyz and range-differences
	@param c Optional. Speed of sound in m/s, defaults to 343.0 m/s.
	@return flattened_output vector of vectors holding 4 entries: x, y, z, TDOA-residual
	@see mpr2003_optim
	*/
	vector<double> output(8);
	vector<double> sub_vect(4);

	vector<vector<double>> flattened_output;
	for (int i = 0; i < all_inputs.size(); i++){
		output = mpr2003_optim(all_inputs[i], c);
		if (!isnan(output[0])){
			for (int i=0; i<4; i++){
				sub_vect[i] = output[i];
			}
		flattened_output.push_back(sub_vect);
		}
		if (!isnan(output[4])){
			for (int i=0; i<4; i++){
				sub_vect[i] = output[i+4];
			}
		flattened_output.push_back(sub_vect);			
		}

	}
	return flattened_output;
	}

/*int main(){
	
	VectorXd input1(15), input2(15);
	input1 << 0,0,1.0,
						  0,1.0,0,
						  1.0,0,0,
						  1.0,1.2,0,
						   -0.8959416468645998,
						  0.0,
						-1.1891568622830988;
	
	input2 << 0,0,1.0,
						  0,1.0,0,
						  1.0,0,0,
						  1.0,1.2,0,
						   0.2279981273412348,
						  0.9481513817965102,
						0.36372892194093875;
	vector<double> mvect = to_vectdouble(input1);
	vector<double> mvect2 = to_vectdouble(input2);
	vector<vector<double>> multi_input;
	multi_input.push_back(mvect);
	multi_input.push_back(mvect2);
	
	
	
	//cout << "mic array \n " << mic_array.transpose() << endl;
	vector<double> oo;
	oo = mpr2003_optim(mvect);
	for (auto ii : oo){
		cout << ii << ", " ;
	}
	cout << "" << endl;
	
	vector<vector<double>> many_out  = many_mpr2003_optim(multi_input);
	for (auto ii:many_out){
		for (auto kk : ii){
			cout << kk << ", " ;
			}
		cout << "" << endl;
	}

}*/
