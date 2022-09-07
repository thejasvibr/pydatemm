/*Wrapper that implements set operations.
*/
#include <algorithm>
#include <set>
#include <vector>
#include <iostream>

/*template<typename T>
struct UnorderedVectorSet {
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    std::vector<T> _data;

    UnorderedVectorSet() = default;
    UnorderedVectorSet(const std::initializer_list<T>& init) : _data(init) {}
    UnorderedVectorSet(UnorderedVectorSet&&) = default;
    UnorderedVectorSet(const UnorderedVectorSet&) = default;
    UnorderedVectorSet(std::vector<T>&& vic) : _data(std::move(vic)) {}

    UnorderedVectorSet& operator=(const UnorderedVectorSet&) = default;

    template<typename K>
        requires std::convertible_to<K, T>
    void insert(const K& value) {
        if (!this->contains(value))
            _data.emplace_back(value);
    }

    template<typename K>
        requires std::convertible_to<K, T>
    bool contains(const K& value) const {
        return std::find(_data.begin(), _data.end(), value) != _data.end();
    }

    template<typename K>
        requires std::convertible_to<K, T>
    auto find(const K& value) const {
        return std::find(_data.begin(), _data.end(), value);
    }

    void clear() {
        _data.clear();
    }

    template<typename Iterator>
    void insert(Iterator start, Iterator end) {
        for (auto it = start; it != end; ++it)
            insert(*it);
    }
    
    template<typename Iterator>
    inline auto erase(Iterator it) {
        return _data.erase(it);
    }
    
    template<class... Args>
    constexpr typename std::vector<T>::reference emplace(Args&&... args) {
        return _data.emplace_back(std::forward<Args>(args)...);
    }

    auto begin() { return _data.begin(); }
    auto end() { return _data.end(); }
    auto begin() const { return _data.begin(); }
    auto end() const { return _data.end(); }
    size_t size() const { return _data.size(); }
    bool empty() const { return _data.empty(); }
};*/

std::set<int> diff_set(const std::set<int>& A, const std::set<int>& B){
	std::set<int> difference;
	std::set_difference(A.begin(), A.end(), B.begin(), B.end(), 
		std::inserter(difference, difference.end()));
	return difference;
}

std::set<int> union_set(const std::set<int>& A, const std::set<int>& B){
	std::set<int> unionset;
	std::set_union(A.begin(), A.end(), B.begin(), B.end(),
		  std::inserter(unionset, unionset.end()));
	return unionset;
}


std::vector<int> diff_set_vect(const std::vector<int>& A, const std::vector<int>& B){
	std::set<int> Aset,Bset;
	std::set<int> diff_result_set;
	for (auto j : A){Aset.insert(j);}
	for (auto k : B){Bset.insert(k);}
	std::vector<int> difference;
	diff_result_set = diff_set(Aset, Bset);
	for (auto i : diff_result_set){difference.push_back(i);}
	return difference;
}

std::vector<int> union_set_vect(const std::vector<int>& A, const std::vector<int>& B){
	std::set<int> Aset,Bset;
	std::set<int> union_set_out;
	for (auto j : A){Aset.insert(j);}
	for (auto k : B){Bset.insert(k);}
	std::vector<int> unionvect;
	union_set_out = union_set(Aset, Bset);
	for (auto i : union_set_out){unionvect.push_back(i);}
	return unionvect;
}

// NOT used as diff_set_vect_v2 is much slower than diff_set_vect (1.61 times slower)>
std::vector<int> diff_set_vect_v2(const std::vector<int>& A, const std::vector<int>& B){
	/*Checks to see which elements of A are unique, and not present in B. 
	>>> diff_set_vect_v2({1,2,3}, {1,3,9})
	Output:
	>>> {2}
	*/
	std::vector<int> difference;
	bool a_in_B;
	
	if (B.empty()){
		for (auto a : A){difference.push_back(a);}
		return difference;
	}
	else if (A.empty()){
		return difference;
	}
	else{
		for (auto a : A){
			for (auto b : B){
			// https://stackoverflow.com/a/5998594/4955732
			a_in_B = std::find(B.begin(), B.end(), a) != B.end();
			if (a_in_B){
				difference.push_back(a);
				}
			}
		}		
	}
	return difference;
}