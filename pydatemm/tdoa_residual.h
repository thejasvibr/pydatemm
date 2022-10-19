/*
Calculates the TDOA residuals given the array geometry and
predicted sources - and then calculates the prediction-observation
residual. 
*/
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::seq;


double euclidean_distance(VectorXd A, VectorXd B){
	double distance;
	distance = (A-B).norm();
	return distance;
	}

double residual_tdoa_error(VectorXd d, const VectorXd source, const MatrixXd arraygeom, double c=343.0){
	int nchannels = arraygeom.rows();
	VectorXd distmat(nchannels);
	VectorXd pred_d(nchannels-1);
	double tdoa_resid;
	for (int i=0; i<nchannels; i++){
		distmat(i) = euclidean_distance(source, arraygeom.row(i));
	}
	//predicted range difference
	VectorXd ref_dist(nchannels-1);
	ref_dist = VectorXd::Ones(nchannels-1)*distmat(0);
	pred_d = distmat.tail(nchannels-1)-ref_dist;
	
	pred_d = pred_d/c; // convert to seconds delay
	d = d/c;
	tdoa_resid = euclidean_distance(pred_d, d);
	tdoa_resid = tdoa_resid/std::sqrt((double)nchannels);
	return tdoa_resid;
	}
