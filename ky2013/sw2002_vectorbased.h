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
#define EIGEN_DONT_PARALLELIZE

	
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
	// Whichever solution results in the lower TDE residual is returned
	if (residuals[0] < residuals[1]){
		return solution1;
	}
	else if (residuals[1] < residuals[0]){
		return solution2;
	} else if (residuals[0] == residuals[1]){
		//std::cout << "Rangediff "<< rangediff.transpose() << std::endl;
		//throw std::invalid_argument( "BOTH RESIDUALS ARE EQUAL!!" );
		// BAD PRACTICE - BUT PUTTING IT NOW SO MY CODE CALLS DON'T ALWAYS CRASH COMPLETELY!
		return to_VXd(vector<double>{-999,-999,-999});
	}
	}


vector<double> sw_matrix_optim(const vector<double> &mic_ntde_raw, const double &c=343.0){
	/*
	
	mic_ntde is 1D vector<double> with nmics*3 + Nmics-1 entries. 
	The entries are organised so: m0_x.1, m0_y.1, m0_z.1,..mNmics_x.1,mNmics_y.1, mNmics_z.1, D10...DNmics0
	
	Where D is the RANGE DIFFERENCE!!!
	
	*/
	//std::cout<<"Miaow "<< std::endl;
	int nmics = get_nmics(mic_ntde_raw);
	Vector3d best_solution;
	VectorXd mic_ntde_raw_vx = to_VXd(mic_ntde_raw);
	//std::cout << "\n" << std::endl;
	//std::cout << std::setprecision(15) << "mic_ntde_raw " << mic_ntde_raw_vx << std::endl;
	//std::cout << "\n" << std::endl;
	VectorXd mic_ntde = to_VXd(mic_ntde_raw);
	VectorXd solutions_vx(6);
	vector<double> solution(3);
	double a1,a2,a3; 
    double a_quad,b_quad, c_quad;
    double t_soln1, t_soln2;
    VectorXd b(nmics-1);
    VectorXd f(nmics-1);
    VectorXd g(nmics-1);
    VectorXd tau(nmics-1);
	VectorXd s1(3),s2(3);
    long long int position_inds = nmics*3;
	VectorXd mic0 = mic_ntde.head(3);
	tau = mic_ntde.tail(nmics-1)/c;
	MatrixXd R(nmics-1,3);
	MatrixXd R_inv(3, nmics-1);
	ArithmeticSequence< long long int, long long int, long long int > starts = seq(3, position_inds-3, 3);
	ArithmeticSequence< long long int, long long int, long long int > stops = seq(5, position_inds-1, 3);
	for (long long int i=0; i<starts.size(); i++){
		mic_ntde(seq(starts[i],stops[i])) +=  -mic0;
		}
	R = mic_ntde(seq(3,position_inds-1)).reshaped(3,nmics-1).transpose();
	
	MatrixXd Eye(R.rows(),R.rows());
    Eye = MatrixXd::Zero(R.rows(), R.rows());
    Eye.diagonal() = VectorXd::Ones(R.rows());
	
    //R_inv = R.colPivHouseholderQr().solve(Eye);
	//double epsilon = std::numeric_limits<double>::epsilon();
	//acobiSVD<MatrixXd> svd; // thanks https://gist.github.com/pshriwise/67c2ae78e5db3831da38390a8b2a209f
	//R_inv = svd.compute(R, ComputeThinU | ComputeThinV).solve(Eye); // thanks https://stackoverflow.com/a/72753193/4955732
	//std::cout << R_inv << std::endl; 
	R_inv = R.fullPivHouseholderQr().solve(Eye);
	//R_inv = R.completeOrthogonalDecomposition().pseudoInverse();
    //R_inv = R.jacobiSvd().solve(Eye);
	//R_inv = pseudoInverse(R);
	
	for (int i=0; i < nmics-1; i++){
	b(i) = pow(R.row(i).norm(),2) - pow(c*tau(i),2);
	f(i) = (c*c)*tau(i);
	g(i) = 0.5*(c*c-c*c);  
  	}
    a1 = (R_inv*b).transpose()*(R_inv*b);
    a2 = (R_inv*b).transpose()*(R_inv*f);
	//std::cout << "\n \n Rinv*f: " << R_inv*f << std::endl;
    a3 = (R_inv*f).transpose()*(R_inv*f);
	//std::cout << "many_u:" << mic_ntde_raw_vx.head(15) << std::endl;
	//std::cout << "Rinv " << R_inv << std::endl;
	//std::cout<< "a1,2,3 "<< a1<< ", "<<a2<< ", "<<a3<<std::endl; 
    a_quad = a3 - pow(c,2);
    b_quad = -1.0*a2;
    c_quad = a1/4.0;
	//std::cout << "a_quad" << a_quad << " b_quad " << b_quad << "c_quad" << c_quad << std::endl;	
	//std::cout << "yy_pt1: " << pow(b_quad,2)<< " yy_pt2: " <<  4*a_quad*c_quad << std::endl;
	//std::cout << "yy_pt1-pt2: " << pow(b_quad,2)-4*a_quad*c_quad << std::endl;
	//std::cout<< "Potential: " << std::setprecision(12) << yy << ",  "<< zz << "\n" <<std::endl;
	
    t_soln1 = (-b_quad + sqrt(pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad);
    t_soln2 = (-b_quad - sqrt(pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad);	
	//std::cout << a_quad << ", "<< b_quad << ", "<<c_quad << ", "<< t_soln1 << ", " << t_soln2 << std::endl;
    solutions_vx(seq(0,2)) = R_inv*b*0.5 - (R_inv*f)*t_soln1;
	solutions_vx(seq(0,2)) += mic0;
    solutions_vx(seq(3,5)) = R_inv*b*0.5 - (R_inv*f)*t_soln2;
	solutions_vx(seq(3,5)) += mic0;
	//std::cout << "Both Solutions" << solutions_vx << std::endl;
	best_solution = choose_correct_solution(solutions_vx, tau*c,mic_ntde_raw_vx.head(nmics*3));
	solution = to_vectdouble(best_solution);
	return solution;
}

vector<vector<double>> pll_sw_optim(const vector<vector<double>> &all_inputs, const int &num_cores, double c=343.0){
	/*
	The non-block based parallel implementation. Lets OMP do all the chunking instead of doing it explicitly. 
	*/

	vector<vector<double>> flattened_output(all_inputs.size());
	// Now run the parallelisable code
	#pragma omp parallel for
	for (int i = 0; i < all_inputs.size(); i++){
		flattened_output[i] = sw_matrix_optim(all_inputs[i], c);
		}

	return flattened_output;
					}


/*int main(){
	std::cout << "starting" << std::endl;
	
	std::vector<double> qq {3.79879879879879,-1.11611611611611,-3.83883883883883,
								-0.745745745745745,-0.525525525525525,4.12912912912912,
								0.765765765765765,1.59659659659659,1.18618618618618,
								-2.57757757757757,0.125125125125125,4.88988988988988,
								-1.28628628628628,-4.78978978978978,-2.37737737737737,
								-0.0905304146393017,-2.31630692205105,-0.244370170276845,-0.270533433265002
								};
								
	 std::cout << " Nmics " << get_nmics(qq) << std::endl;						
	//VectorXd mictde = to_VXd(qq);
	/*
	int n_mics = 5;
	vector<double> output;
	auto start = chrono::steady_clock::now();
	output = sw_matrix_optim(qq, n_mics);
	auto stop = chrono::steady_clock::now();
	double durn1 = chrono::duration_cast<chrono::microseconds>(stop - start).count();
	
	for (auto ii : output){
		std::cout << ii  << std::endl;
	}
	
	// Now run the parallelised version 
	int nruns = 10000;
	vector<vector<double>> block_in(nruns);
	vector<vector<double>> pll_out;
	vector<int> block_nmics(block_in.size());
	
	std::cout << block_in.size() << std::endl;
	for (int i=0; i < block_in.size(); i++){
		block_in[i] = qq;
		block_nmics[i] = n_mics;
	}
	// run the whole code without parallelism
	std::cout << "Serial run starting... " << std::endl;
	auto start1 = chrono::steady_clock::now(); 
	vector<vector<double>> serial_out(nruns);
	for (int i=0; i<nruns; i++){
		serial_out[i] = sw_matrix_optim(qq, n_mics);
	}
	auto stop1 = chrono::steady_clock::now();
	durn1 = chrono::duration_cast<chrono::microseconds>(stop1-start1).count();
	std::cout << durn1 << " Serial s"<< std::endl;

	// Now finally try to run the actual pll function
	std::cout << "Parallel run starting... " << std::endl;
	auto start2 = chrono::steady_clock::now();
	pll_out = pll_sw_optim(block_in, block_nmics, 8, 343.0);
	auto stop2 = chrono::steady_clock::now();
	auto durn2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2).count();
	std::cout << durn2 << " FN pll s"<< std::endl;
	
	std::cout << "Obtained speedup: " << durn1/durn2 << std::endl;
	
	VectorXd corr_soln;
	//corr_soln = choose_correct_solution(
	MatrixXd reshaped = to_VXd(qq).head(15).reshaped(3,5).transpose();
	VectorXd correct_soln;
	VectorXd bothsoln, tau, micxyz;
	bothsoln = to_VXd(serial_out[0]);
	std::cout << "Both Solns " << bothsoln << std::endl;
	
	return 0;}*/
