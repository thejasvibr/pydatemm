// Reads a csv file and begins performing the CombineAll routine. 
#include <iostream> 
#include <fstream>
#include <vector>
#include <cmath>
#include "combineall.h"
#include "Timer.h"

std::vector<std::vector<int> > create_acc_graph(std::string filepath){
	
	long double num_entries = 0;
	int num_rows;
	unsigned long long count = 0;
	std::vector<int> temp_row;
	std::vector<std::vector<int> > acc;
	std::vector<int> data;
	

	std::ifstream input(filepath);
	// get number of entries
	std::string line;
	if (input.is_open()){
		while (!input.eof()){
			std::getline(input, line);
			count++;
		}
		input.close();
	}

	num_entries = count-1;
	std::cout << num_entries << std::endl;

	num_rows = (int)std::sqrt(num_entries);
	std::cout << "CCG will be of size: " << num_rows << "x" << num_rows << std::endl;

	auto numentries = (unsigned long long)num_entries;
	data.reserve(numentries);
	std::ifstream input2(filepath);
	for (auto i = 0; i < numentries; i++) {
        input2 >> data[i];
        }

	// create acc graph
	for (int i=0; i<num_rows; i++){
		for (int j=0; j<num_rows; j++){
			
			if (i<1){
				//acc.push_back(data[j]);
				temp_row.push_back(data[j]);
				//std::cout << data[j]<< std::endl;
			}
			else{
				//acc[i].push_back(data[((i*num_rows)+j)]);
				temp_row.push_back(data[((i*num_rows)+j)]);
				//std::cout << data[(i*num_rows)+j]<< std::endl;
			}
		}
		acc.push_back(temp_row);
		temp_row.clear();	
	}
	return acc;
}

void write_combineall_solutions(std::vector<std::set<int>> solutions){
	// Begin writing the data
	int num_solutions = solutions.size();
	std::cout << "Number of solutions is: " << num_solutions << std::endl;
	
	std::ofstream fw("combineall_solutions.csv", std::ofstream::out);
	if (fw.is_open()){
		for (auto soln : solutions){
			for (auto every : soln){
				fw << every << ",";
			}
			fw << std::endl;
		}
	fw.close();
	}
	else{
		std::cout << "Problem with opening file";
	}
	
}


int main(int argc, char* argv[]){
	std::set<int> V_t;
	std::set<int> ll;
	std::set<int> X;
	std::vector<std::set<int> > solutions;
	int num_nodes;
	
	std::cout<<"Reading txt file" <<std::endl;
	std::vector<std::vector<int> > acc = create_acc_graph(argv[1]);
	std::cout<<"...Done reading txt file" <<std::endl;
	
	num_nodes = acc.size();
	//initialise vector set
	for (int k=0; k<num_nodes; k++) {
		V_t.insert(k);
	}
	std::cout << "Starting CombineAll run..." << std::endl;
	Timer timer1;
	solutions = combine_all(acc, V_t, ll, X);
	std::cout << "The run took:" << timer1.elapsed() << std::endl;
	std::cout << "Done with CombineAll run..." << std::endl;
	
	write_combineall_solutions(solutions);

	return 0;
}
