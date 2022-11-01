#include <omp.h>
#include <iostream>
#include <chrono>
//#define EIGEN_DONT_PARALLELIZE
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
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Dynamic;
using Eigen::ArithmeticSequence;
using Eigen::seq;
using Eigen::seqN;
using Eigen::JacobiSVD;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using namespace std;



double rangediff_pair(Vector3d source, int chX, MatrixXd all_mic_posns){
	double ch0_dist, chX_dist;
	double rangediff;
	ch0_dist = (source - VectorXd(all_mic_posns.row(0))).norm();
	chX_dist = (source - VectorXd(all_mic_posns.row(chX))).norm();
	rangediff = chX_dist - ch0_dist;
	return rangediff;
	}


VectorXd choose_correct_solution(VectorXd both_solutions, const VectorXd &rangediff, VectorXd all_micxyz){
	Vector3d solution1, solution2;
	MatrixXd array_geom;
	solution1 = both_solutions.head(3);
	solution2 = both_solutions.tail(3);
	vector<double> calculated_tdes(2);
	vector<double> residuals(2);
	// Get predicted TDEs from predicted source locations
	int nmics = all_micxyz.size()/3;
	array_geom = all_micxyz.reshaped(3,nmics).transpose();

	calculated_tdes[0] = rangediff_pair(solution1, 4, array_geom);
	calculated_tdes[1] = rangediff_pair(solution2, 4, array_geom);
    
	residuals[0] = abs(rangediff[3] - calculated_tdes[0]);
	residuals[1] = abs(rangediff[3] - calculated_tdes[1]);
	if (isnan(residuals[0]) || isnan(residuals[1])){
    	return Vector3d({-999, -999, -999});
    	}
	
	// Whichever solution results in the lower TDE residual is returned
	if (residuals[0] < residuals[1]){
		return solution1;
	}
	else if (residuals[1] < residuals[0]){
		return solution2;
    	} 
	}

vector<double> sw_matrix_optim(const vector<double> &mic_ntde_raw, const double &c=343.0){
	/*
	
	mic_ntde is 1D vector<double> with nmics*3 + Nmics-1 entries. 
	The entries are organised so: m0_x.1, m0_y.1, m0_z.1,..mNmics_x.1,mNmics_y.1, mNmics_z.1, D10...DNmics0
	
	Where D is the RANGE DIFFERENCE!!!
	
	*/
	VectorXd intermediate_out(4);
	int nmics = get_nmics(mic_ntde_raw);
	Vector3d best_solution;
	VectorXd mic_ntde_vx_raw = to_VXd(mic_ntde_raw); // without any subtraction

	VectorXd mic_ntde = to_VXd(mic_ntde_raw);
	VectorXd solutions_vx(6);
	vector<double> solution(4);
	double a1,a2,a3; 
    double a_quad,b_quad, c_quad;
    double t_soln1, t_soln2;
	double tdoa_resid;
    VectorXd b(nmics-1);
    VectorXd f(nmics-1);
    VectorXd g(nmics-1);
    VectorXd tau(nmics-1);
	VectorXd s1(3),s2(3);

    int position_inds = nmics*3;  
   	VectorXd mic0 = mic_ntde.head(3);
   	tau = mic_ntde.tail(nmics-1)/c;
   	MatrixXd R(nmics-1,3);
   	MatrixXd R_inv(3, nmics-1);
   	ArithmeticSequence starts = seq(3, position_inds-3, 3);
   	ArithmeticSequence stops = seq(5, position_inds-1, 3);
   	for (int i=0; i<starts.size(); i++){
   		mic_ntde(seq(starts[i],stops[i])) +=  -mic0;
   		}
    		
	R = mic_ntde(seq(3,position_inds-1)).reshaped(3,nmics-1).transpose();
	MatrixXd Eye(R.rows(),R.rows());
    Eye = MatrixXd::Zero(R.rows(), R.rows());
    Eye.diagonal() = VectorXd::Ones(R.rows());
	
    R_inv = R.fullPivHouseholderQr().solve(Eye);
	
	for (int i=0; i < nmics-1; i++){
		b(i) = pow(R.row(i).norm(),2) - pow(c*tau(i),2);
		f(i) = (c*c)*tau(i);
		g(i) = 0.5*(c*c-c*c);  
  	}
    a1 = (R_inv*b).transpose()*(R_inv*b);
    a2 = (R_inv*b).transpose()*(R_inv*f);
    a3 = (R_inv*f).transpose()*(R_inv*f);
	a_quad = a3 - pow(c,2);
    b_quad = -1.0*a2;
    c_quad = a1/4.0;
    t_soln1 = (-b_quad + sqrt(pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad);
    t_soln2 = (-b_quad - sqrt(pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad);	

	solutions_vx(seq(0,2)) = R_inv*b*0.5 - (R_inv*f)*t_soln1;
	solutions_vx(seq(0,2)) += mic0;
    solutions_vx(seq(3,5)) = R_inv*b*0.5 - (R_inv*f)*t_soln2;
	solutions_vx(seq(3,5)) += mic0;

	best_solution = choose_correct_solution(solutions_vx, tau*c, mic_ntde_vx_raw.head(nmics*3));

	MatrixXd arraygeom(nmics,3);
	arraygeom = mic_ntde_vx_raw(seq(0,3*nmics - 1)).reshaped(3,nmics).transpose();
	intermediate_out(3) = residual_tdoa_error(mic_ntde_vx_raw.tail(nmics-1), best_solution, arraygeom, c);
	intermediate_out.head(3) = best_solution;

	solution = to_vectdouble(intermediate_out);
	return solution;
}

vector<vector<double>> pll_sw_optim(const vector<vector<double>> &all_inputs, const int &num_cores, double c=343.0){
	/*
	The non-block based parallel implementation. Lets OMP do all the chunking instead of doing it explicitly. 
	*/

	vector<vector<double>> flattened_output(all_inputs.size());
	// Now run the parallelisable code
	omp_set_num_threads(num_cores);
	#pragma omp parallel for 
	for (int i = 0; i < all_inputs.size(); i++){
		flattened_output[i] = sw_matrix_optim(all_inputs[i], c);
		}

	return flattened_output;
					}