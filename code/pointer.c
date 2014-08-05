#include "mex.h"

/* Returns the address of the variable and the address of its value.
 * C-MEX for MATLAB. Compile with "mex pointer.c" (ignore the warnings)
 * Inspired by Julia's "pointer".
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char retStr[128];
    double *ptr;

    if (nrhs != 1) {
	mexErrMsgTxt("One and only one argument is required.");
    }

    ptr = mxGetPr(prhs[0]);

    /* mexPrintf("[%x => [%x]]\n", prhs[0], &ptr[0]); */
    sprintf(retStr, "[0x%x => [0x%x]]", prhs[0], &ptr[0]);

    plhs[0] = mxCreateString(retStr);
}
