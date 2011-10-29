#include "SimpleSparseMatrix.h"

#include <cstdio>

int main(int nargs, char** args)
{

	SimpleSparseMatrix<float> ssm(5, 4);

	ssm.set(0, 0, 3.5);
	ssm.set(0, 3, 2.5);
	ssm.set(1, 2, 6.2);
	ssm.set(1, 0, 8.1);
	ssm.set(4, 3, 6.89);

	vector<ssm_index*>* nz = ssm.nonzero_indicies();
	ssm_index* indx;
	int i, j;
	float val = 0;
	for (int k = 0; k < nz->size(); k++) {
		indx = (*nz)[k];
		i = indx[0];
		j = indx[1];
		val = ssm.get(i, j);

		printf("ssm[%d][%d]=%f\n", i, j, val);
	}

	return 0;

}
