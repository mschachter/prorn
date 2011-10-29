#ifndef _SS_MATRIX_H
#define	_SS_MATRIX_H

#include <map>
#include <cmath>
#include <vector>

using namespace std;

typedef unsigned int ssm_index;

template <class T>
class SimpleSparseMatrix
{
public:
    typedef map<ssm_index, T> mat_map;

    SimpleSparseMatrix(ssm_index i, ssm_index j)
    {
    	nrows=i;
    	ncols=j;
    }

    inline
    T get(ssm_index i, ssm_index j)
    {
        if (nrows <= i || ncols <= j) throw;
        ssm_index key = get_index(i, j);
        return value_map[key];
    }

    inline
    void set(ssm_index i, ssm_index j, T val)
    {
        if (nrows <= i || ncols <= j) throw;
        ssm_index key = get_index(i, j);
        value_map[key] = val;
    }

    inline
    vector<ssm_index*>* nonzero_indicies()
    {
    	vector<ssm_index*>* ilist = new vector<ssm_index*>();

    	typename mat_map::iterator iter = value_map.begin();
    	ssm_index k;
    	int i, j;
    	while (iter != value_map.end()) {
    		k = iter->first;
    		j = k % ncols;
    		i = (int) ((k - j) / ncols);
    		ssm_index* indx = new ssm_index[2];
    		indx[0] = i;
    		indx[1] = j;
    		ilist->push_back(indx);
    		iter++;
    	}

    	return ilist;
    }


protected:
    SimpleSparseMatrix(){}
    ssm_index get_index(ssm_index i, ssm_index j) {    return i*ncols + j; }

private:
    mat_map value_map;
    ssm_index nrows;
    ssm_index ncols;
};

#endif


