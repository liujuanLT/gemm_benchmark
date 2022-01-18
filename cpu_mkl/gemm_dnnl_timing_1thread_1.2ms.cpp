
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "dnnl.hpp"
#include "mkl.h"

using namespace dnnl;
using namespace std;

namespace {

// void init_vector(std::vector<float> &v) {
//     std::mt19937 gen;
//     std::uniform_real_distribution<float> u(-1, 1);

//     for (auto &e : v)
//         e = u(gen);
// }

void init_vector(std::vector<float> &v) {
    for (int i = 0; i < v.size(); ++i) {
        v[i] = i+1;
    }
}


} // namespace

void print_matrix(const vector<float>& mat, int m, int n) {
    for (int irow = 0; irow < min(m, 10); ++ irow) {
        for (int icol = 0; icol < min(n, 10); ++ icol) {
            std::cout << mat[n*irow + icol] << " " ;
        }
        std::cout << std::endl;
    }
}

int number_of_runs_warmup = 1000;
int number_of_runs = 1000;
float fixed_beta = 0.f;

void sgemm_with_params(char transA, char transB, int64_t M,
        int64_t N, int64_t K, float alpha, float beta) {
    if (beta != fixed_beta)
        throw std::logic_error("Run-time beta is not yet supported.");

    double s_initial, s_elapsed;
  
    // Allocate and initialize matrices
    std::vector<float> A(M * K);
    init_vector(A);

    std::vector<float> B(K * N);
    init_vector(B);

    std::vector<float> C_sgemm(M * N);
    init_vector(C_sgemm);

    std::vector<float> C_dynamic_matmul = C_sgemm;
    std::vector<float> C_static_matmul = C_sgemm;

    // Prepare leading dimensions
    int64_t lda = tolower(transA) == 'n' ? K : M;
    int64_t ldb = tolower(transB) == 'n' ? N : K;
    int64_t ldc = N;

    // warm up
    for (int run = 0; run < number_of_runs_warmup; ++run)
      dnnl_sgemm(transA, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb,
                  beta, C_sgemm.data(), ldc);
    s_initial = dsecnd(); // warm up the timing function
                
    s_initial = dsecnd();
    // 1. Execute sgemm
    for (int run = 0; run < number_of_runs; ++run)
        dnnl_sgemm(transA, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb,
                beta, C_sgemm.data(), ldc);
    
    s_elapsed = (dsecnd() - s_initial) / number_of_runs;

    cout << "A:" << endl;
    print_matrix(A, M, K);
    cout << "B:" << endl;
    print_matrix(B, K, N);
    cout << "C:" << endl;
    print_matrix(C_sgemm, M, N);
    
    printf ( " time for loop %d : %.5f ms using %d threads \n\n", number_of_runs, (s_elapsed * 1000), 1);
}

void sgemm() {
    // sgemm_and_matmul_with_params('N', 'T', 10, 20, 30, 1.1f, fixed_beta);
    sgemm_with_params('N', 'N', 1024, 1024, 1024, 1.0f, fixed_beta);
}

int main(int argc, char **argv) {
    sgemm();
}
