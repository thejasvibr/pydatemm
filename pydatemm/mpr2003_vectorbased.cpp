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
using Eigen::Vector3d, Eigen::Vector2d;
using Eigen::VectorXd;
using Eigen::ArithmeticSequence;
using Eigen::seq;
using namespace std;


Vector2d  solve_eqn_22(VectorXd a, VectorXd b){
	Vector2d Rs12;
	
	Rs12	<< 1,2;
	double term1, term2, bsquare_term, term2_ii;
	double term2_i;
	
	term1 = (a.array()*b.array()).sum();
	term2_i = pow(term1, 2);
	bsquare_term = b.array().pow(2).sum() - 1 ;
	term2_ii = bsquare_term*(a.array().pow(2).sum());
	term2 = sqrt(term2_i - term2_ii);
	cout << "term2  " << term2 << endl;
	
	return Rs12;
}

vector<double> mpr2003_optim(const vector<double> &mic_ntde_raw, const double &c=343.0){
    int nmics = get_nmics(mic_ntde_raw);
	cout << "num mics is " << nmics << endl;
	VectorXd mic_ntde_vx_raw = to_VXd(mic_ntde_raw); // without any subtraction
	VectorXd mic_ntde = to_VXd(mic_ntde_raw);
	MatrixXd S(nmics-1,3), invS_t_S(nmics-1,3);
	MatrixXd S_t(nmics-1,3), inv_StS_St(nmics-1,3);
	VectorXd di(nmics-1),z(nmics-1),a(nmics-1), b(nmics-1);
	Vector2d Rs_12;

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
    cout << "z" << z << endl;
	z *= 0.5;
	//eqn. 17 - without the weighting matrix R
	S_t = S.replicate(1,1).transpose();
	cout << "S_t \n" << S_t << endl;
	invS_t_S = (S_t*S).inverse();
	inv_StS_St = invS_t_S*S_t;
	cout << "inv_StS_St \n" << inv_StS_St << endl;
	a = inv_StS_St*z;
	// eqn. 22
	b = inv_StS_St*di;
    
	//eqn. 22
	Rs_12 = solve_eqn_22(a, b);
 
    /*# eqn. 18
    b = np.dot(inv_StS_St, di);
    # eqn. 22
    Rs_12= solve_eqn_22(a, b);
    # substitute Rs into eqn. 19
    //xs = choose_correct_mpr_solutions(mic_array, Rs_12, (a,b), di);
    return xs;*/
	vector<double> out = {0,1.5,2.4};
	return out;
}



/*
def solve_eqn_22(a,b):
    '''
    Implements Equation 22 in MPR 2003. 

    Parameters
    ----------
    a,b : (3,) np.arrays
    
    Returns
    -------
    Rs12 : (2,) np.array
        Array holding Rs1 and Rs2 - the two range estimates arising from 
        the quadratic equation.
    '''
    a1, a2, a3 = a
    b1, b2, b3 = b
    term1 = a1*b1 + a2*b2 + a3*b3
    # split the numerator
    term2_i = term1**2
    bsquare_term = (b1**2+ b2**2 + b3**2) - 1 

    term2_ii = bsquare_term*(a1**2 + a2**2 + a3**2)
    term2 = np.sqrt(term2_i - term2_ii)
    denominator = bsquare_term
    numerator1 = term1 + term2
    numerator2 = term1 - term2
    Rs12 = np.array([numerator1/denominator, numerator2/denominator])
    return Rs12


*/

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
