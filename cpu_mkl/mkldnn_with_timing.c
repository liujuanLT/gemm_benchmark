#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 10

int main(int argc, char *argv[])
{
    double *A, *B, *C;
    int m, n, p, i, r;
    double alpha, beta;
    double s_initial, s_elapsed;
    if (argc == 4){
      m = atoi(argv[1]);
      n = atoi(argv[2]);
      k = atoi(argv[3]);
      printf(" m %i n %i k %i",m,n,k);
    }
    else{
      m=2000,k=200,n=1000;
    }

    // m = 2000, p = 200, n = 1000;

    printf ("\n This example measures performance of Intel(R) MKL function dgemm \n"
            " computing real matrix C=alpha*A*B+beta*C, where A, B, and C \n"
            " are matrices and alpha and beta are double precision scalars\n\n");

    
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, p, p, n);
    alpha = 1.0; beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    B = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*p); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (p*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    printf (" Making the first run of matrix product using Intel(R) MKL dgemm function \n"
            " via CBLAS interface to get stable run time measurements \n\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, p, alpha, A, p, B, n, beta, C, n);

    printf (" Measuring performance of matrix product using Intel(R) MKL dgemm function \n"
            " via CBLAS interface \n\n");
    s_initial = dsecnd();
    for (r = 0; r < LOOP_COUNT; r++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, p, alpha, A, p, B, n, beta, C, n);
    }
    s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;

    printf (" == Matrix multiplication using Intel(R) MKL dgemm completed == \n"
            " == at %.5f milliseconds == \n\n", (s_elapsed * 1000));

    printf (" Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    if (s_elapsed < 0.9/LOOP_COUNT) {
        s_elapsed=1.0/LOOP_COUNT/s_elapsed;
        i=(int)(s_elapsed*LOOP_COUNT)+1;
        printf(" It is highly recommended to define LOOP_COUNT for this example on your \n"
               " computer as %i to have total execution time about 1 second for reliability \n"
               " of measurements\n\n", i);
    }

    printf (" Example completed. \n\n");
    return 0;
}

