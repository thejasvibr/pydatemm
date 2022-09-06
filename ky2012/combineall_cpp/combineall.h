/*
CombineAll is a routine which solves a 'compatibility' problem.
Essentially there are multiple vertices in a graph that are either compatible
or not with each other. CombineAll finds all possible combinations of 
vertices that are compatible with each other. The compatibility/conflict between
vertices is represented as a 2D matrix with +1 for compatibility and -1 for 
conflict (and 0's to indicate NAs). 

Reference
---------
* Kreissig & Yang 2013, Fast and reliable TDOA assignment in multi-source reverberant environments, ICASSP 2013
  https://ieeexplore.ieee.org/abstract/document/6637668?casa_token=3oKOQUJRuWQAAAAA:JNbwI-gf0m0ozfAKbAQJzblq8qE-NPTJ49hgJILMxG_2ZM9MJOt4PQOvPEQn9TXJZSzD_ON6YA
*/

#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include "set_operations.h" // diff_set and union_set set operations defined here

/**
  Gets compatible neighbours 

  @param Acc  The Compatibility-Conflict graph. 
  @param V_t  The set of vertices available, (V tilde). 
  @param l    The current solution set.
  @return Nvl Compatible solution set. 
  **/
std::set<int> get_Nvl(std::vector<std::vector<int> > &Acc, const std::set<int>& V_t, const std::set<int>& l){
		int index;
		std::set<int> Nvl;
		if (l.empty()==true){
			for (auto ii : V_t){
				Nvl.insert(ii);
				}
		}
		else{
			for (int v : V_t){
				for (int u : l){
					if (Acc[v][u]==1){
						Nvl.insert(v);
					}
					else if(Acc[v][u]==-1){
						/*find where v is in Nvl
						index = searchResult(Nvl, v);
						and eliminate it from Nvl
						Nvl.erase(Nvl.begin()+index);*/
						Nvl.erase(v);
					}
				}
			}
			
			}
	return Nvl;
}


/**
  Gets INcompatible neighbours 

  @param Acc  The Compatibility-Conflict graph. 
  @param V_t  The set of vertices available, (V tilde). 
  @param l    The current solution set.
  @return N_not_vl Inompatible vertices.
  **/
std::set<int> get_not_Nvl(std::vector<std::vector<int> > &Acc, const std::set<int>& V_t, const std::set<int>& l){
		//int index;
		std::set<int> N_not_vl;
		if (l.empty()==true){
			return N_not_vl;}
		else{
			for (int v : V_t){
				for (int u : l){
					if (Acc[v][u] == -1){
						N_not_vl.insert(v);
					}
					else if(Acc[v][u] == 1){
						/*find where v is in Nvl
						index = searchResult(N_not_vl, v);*/
						/*and eliminate it from Nvl*/
						N_not_vl.erase(v);
					}
				}
			}
			
			}
	return N_not_vl;
}


/**
  Finds multiple compatible solutions where nodes don't conflict each other.

  @param Acc  The Compatibility-Conflict graph. 
  @param V_t  The set of vertices available 
  @param l    The current solution set.
  @param X    Already visited vertices.
  @solutions_l All possible vertex combinations that are compatible with each other.
  **/

std::vector<std::set<int>> combine_all(std::vector<std::vector<int> > &Acc, std::set<int> V, const std::set<int>& l, std::set<int> X){
	std::set<int> Nvl, N_not_vl;
	std::vector<std::set<int> > solutions_l;
	std::set<int> Nvl_wo_X, Vx;
	std::set<int> temp_set, lx;
	std::vector<std::set<int> > current_solution;

	Nvl = get_Nvl(Acc, V, l);
	N_not_vl = get_not_Nvl(Acc, V, l);
	
	if (Nvl.empty()){
		solutions_l.push_back(l);
	}
	else{
		// remove conflicting nodes
		V = diff_set(V, N_not_vl);
		// unvisited compatible neighbours
		Nvl_wo_X = diff_set(Nvl, X);
		for (int n : Nvl_wo_X){
			temp_set = {n};
			Vx = diff_set(V, temp_set);
			lx = union_set(l, temp_set);
			current_solution = combine_all(Acc, Vx, lx, X);
			if (!current_solution.empty()){
				for (std::set each : current_solution){
					solutions_l.push_back(each);
				}

			}
			// build onto set of visited neighbours
			X.insert(n);
		}
	}
	return solutions_l;
}