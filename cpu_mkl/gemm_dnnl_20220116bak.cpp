
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


void init_vector(std::vector<int8_t> &v) {
    for (int i = 0; i < v.size(); ++i) {
        v[i] = i+1;
    }
}

void init_vector(std::vector<int32_t> &v) {
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

void print_matrix(const vector<int8_t>& mat, int m, int n) {
    for (int irow = 0; irow < min(m, 10); ++ irow) {
        for (int icol = 0; icol < min(n, 10); ++ icol) {
            std::cout << (int32_t)mat[n*irow + icol] << " " ;
        }
        std::cout << std::endl;
    }
}

void print_matrix(const vector<int32_t>& mat, int m, int n) {
    for (int irow = 0; irow < min(m, 10); ++ irow) {
        for (int icol = 0; icol < min(n, 10); ++ icol) {
            std::cout << mat[n*irow + icol] << " " ;
        }
        std::cout << std::endl;
    }
}


void sgemm(int64_t M, int64_t N, int64_t K, const int loop_warmup, const int loop_timing) {
    cout <<"M, K, N: " << M << ", " << K << ", " << N << endl;
    double s_initial, s_elapsed;
    
    char transA = 'N';
    char transB = 'N';  // T
    float alpha = 1.0;
    float beta = 0.0f;
    int64_t lda = tolower(transA) == 'n' ? K : M;
    int64_t ldb = tolower(transB) == 'n' ? N : K;
    int64_t ldc = N;
  
    // Allocate and initialize matrices
    std::vector<float> A(M * K);
    init_vector(A);

    std::vector<float> B(K * N);
    init_vector(B);

    std::vector<float> C(M * N);
    init_vector(C);

    
    int max_threads = mkl_get_max_threads();
    // cout << "max_threads: " << max_threads << endl;

    mkl_set_num_threads(max_threads);
    // warm up
    for (int run = 0; run < loop_warmup; ++run)
      dnnl_sgemm(transA, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb,
                  beta, C.data(), ldc);
    s_initial = dsecnd(); // warm up the timing function
                
    s_initial = dsecnd();
    // 1. Execute sgemm
    for (int run = 0; run < loop_timing; ++run)
        dnnl_sgemm(transA, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb,
                beta, C.data(), ldc);
    
    s_elapsed = (dsecnd() - s_initial) / loop_timing;

    //cout << "A:" << endl;
    //print_matrix(A, M, K);
    //cout << "B:" << endl;
    //print_matrix(B, K, N);
    //cout << "C:" << endl;
    //print_matrix(C, M, N);
    
    printf ( "time for loop %d : %.5f ms using %d threads \n", loop_timing, (s_elapsed * 1000), 1);
    printf ( "total time: %.5f s \n", (s_elapsed * 1 * loop_timing));
}


void s8gemm(int64_t M, int64_t N, int64_t K, const int loop_warmup, const int loop_timing) {
    cout <<"M, K, N: " << M << ", " << K << ", " << N << endl;

    double s_initial, s_elapsed;
    
    char transA = 'N';
    char transB = 'N';  // T
    char offsetC = 'F';
    float alpha = 1.0;
    float beta = 0.0f;
    int8_t ao = 0;
    int8_t bo = 0;
    int32_t co = 0;
    int64_t lda = tolower(transA) == 'n' ? K : M;
    int64_t ldb = tolower(transB) == 'n' ? N : K;
    int64_t ldc = N;
  
    // Allocate and initialize matrices
    std::vector<int8_t> A(M * K);
    init_vector(A);

    std::vector<int8_t> B(K * N);
    init_vector(B);

    std::vector<int32_t> C(M * N);
    init_vector(C);

    
    int max_threads = mkl_get_max_threads();
    //cout << "max_threads: " << max_threads << endl;

    mkl_set_num_threads(max_threads);
    // warm up
    for (int run = 0; run < loop_warmup; ++run)
      dnnl_gemm_s8s8s32(transA, transB, offsetC, M, N, K, alpha, A.data(), lda, ao, B.data(), ldb, bo, 
                  beta, C.data(), ldc, &co);
    s_initial = dsecnd(); // warm up the timing function
                
    s_initial = dsecnd();
    // 1. Execute sgemm
    for (int run = 0; run < loop_timing; ++run)
        dnnl_gemm_s8s8s32(transA, transB, offsetC, M, N, K, alpha, A.data(), lda, ao, B.data(), ldb, bo, 
                beta, C.data(), ldc, &co);
    
    s_elapsed = (dsecnd() - s_initial) / loop_timing;

    //cout << "A:" << endl;
    //print_matrix(A, M, K);
    //cout << "B:" << endl;
    //print_matrix(B, K, N);
    //cout << "C:" << endl;
    //print_matrix(C, M, N);
    
    printf ( "time for loop %d : %.5f ms \n", loop_timing, (s_elapsed * 1000));
    printf ( "total time: %.5f s \n", (s_elapsed * 1 * loop_timing));
}

class GEMMARG {
public:
GEMMARG (int64 M_, int64 N_, int64 K_, int loop1, int loop2) :
  M(M_), N(N_), K(K_), loop_warmup(loop1), loop_timing(loop2) 
  {}
  
int64_t M;
int64_t N;
int64_t K;
int loop_warmup;
int loop_timing;
};

void batch_gemm(int method, bool order) {
if (1 == method) {
  /*  vector<GEMMARGS> args ({
    GEMMARG {1024, 1024, 1024, 1E4, 1E4},
    GEMMARG {1, 256, 64, 1E7/2, 1E7/2},
    GEMMARG(64, 256, 64, 1E6/2, 1E6/2),
    GEMMARG(128, 256, 64, 8E5/2, 5E5/2),
    GEMMARG(256, 256, 64, 5E5/2, 5E5/2),
    GEMMARG(512, 256, 64, 500000/2, 500000),
    GEMMARG(1024, 256, 64,500000/2, 500000/2), 
    GEMMARG(2048, 256, 64,500000/2, 500000/2),
    GEMMARG(1, 64, 64, 1E7/2, 1E7/2),
    GEMMARG(1, 128, 128, 5E6/2, 5E6/2), 
    GEMMARG(1, 256, 256, 1E6/2, 1E6/2),
    GEMMARG(1, 512, 512, 1E6/2, 1E6/2),
    GEMMARG(1, 1024, 1024, 5E5/2, 5E5/2),
    GEMMARG {1, 2048, 2048, 2E5/2, 2E5/2}
    });*/
    
   vector<GEMMARGS> args(0);
    //args.push_back(GEMMARG {1024, 1024, 1024, 1E4, 1E4});
    
   /*for (int icase = 0; icase < args.size(); ++ icase) {
      int i = (order) ? icase : args.size() - 1 - icase;
      const GEMMARG& p = args[i];
      sgemm(p.M, p.N, p.K, p.loop_warmup, p.loop_timing);
    }*/
    
 }
 

 /*   sgemm(1024, 1024, 1024, 1E4, 1E4);
    sgemm(1, 256, 64, 1E7/2, 1E7/2); 
    sgemm(64, 256, 64, 1E6/2, 1E6/2); 
    sgemm(128, 256, 64, 8E5/2, 5E5/2);
    sgemm(256, 256, 64, 5E5/2, 5E5/2); 
    sgemm(512, 256, 64, 500000/2, 500000); 
    sgemm(1024, 256, 64,500000/2, 500000/2); 
    sgemm(2048, 256, 64,500000/2, 500000/2);
    sgemm(1, 64, 64, 1E7/2, 1E7/2); 
    sgemm(1, 128, 128, 5E6/2, 5E6/2); 
    sgemm(1, 256, 256, 1E6/2, 1E6/2);
    sgemm(1, 512, 512, 1E6/2, 1E6/2);
    sgemm(1, 1024, 1024, 5E5/2, 5E5/2);
    sgemm(1, 2048, 2048, 2E5/2, 2E5/2);*/
    //s8gemm(1024, 1024, 1024, 3E4, 3E4);
    //s8gemm(1, 256, 64, 1E7, 1E7); 
    //s8gemm(64, 256, 64, 1E6, 1E6); 
    //s8gemm(128, 256, 64, 8E5, 5E5);
    //s8gemm(256, 256, 64, 5E5, 5E5); 
    //s8gemm(512, 256, 64, 500000, 500000); 
    //s8gemm(1024, 256, 64,500000, 500000); 
    //s8gemm(2048, 256, 64,500000, 500000);
    //s8gemm(1, 64, 64, 1E7, 1E7); 
    //s8gemm(1, 128, 128, 5E6, 5E6); 
    //s8gemm(1, 256, 256, 1E6, 1E6);
    //s8gemm(1, 512, 512, 1E6, 1E6);
    //s8gemm(1, 1024, 1024, 5E5, 5E5);
    //s8gemm(1, 2048, 2048, 2E5, 2E5);
}


int main(int argc, char **argv) {
    //cout << "flag " << 9 << endl;
    bool order = true;
    for (int irun = 0; irun < 6; ++ irun) {
      batch_gemm(1, order);
      order = !order;
    }
}
