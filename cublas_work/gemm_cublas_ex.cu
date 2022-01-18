#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <assert.h>
#include "./cmd_line.h"

#ifndef NVCC_ARCHS
#define NVCC_ARCHS 0
#endif

#ifndef CHECK_ARCH
#define CHECK_ARCH 0
#endif

#define LOG_LEVEL 1

using namespace std;

#ifndef DATA_TYPE
#define DATA_TYPE 2  // 0: float, 1: half, 1:int8
#endif

#if DATA_TYPE == 0
    using datatypeA = float;
    using datatypeB = float;
    using datatypeC = float;
#elif DATA_TYPE == 1
    using datatypeA = __half;
    using datatypeB = __half;
    using datatypeC = __half;
#else
    using datatypeA = char;
    using datatypeB = char;
    using datatypeC = int;
#endif

// void init_vector(float* d_data, int len) {
//     // a rough piece of code considering no performance 
//     float* h_data;
//     cudaMallocHost((void**) &h_data, sizeof(float) * len);
//     for (size_t i = 0; i < len; ++ i) {
//         h_data[i] = (float) (i + 1);
//     }
//     cudaMemcpy(d_data, h_data, sizeof(float) * len, cudaMemcpyHostToDevice);
//     cudaFreeHost(h_data);
// }

void init_vector(int* d_data, int len) {
    // a rough piece of code considering no performance 
    int* h_data;
    cudaMallocHost((void**) &h_data, sizeof(int) * len);
    for (size_t i = 0; i < len; ++ i) {
        h_data[i] = (int) (i + 1);
    }
    cudaMemcpy(d_data, h_data, sizeof(int) * len, cudaMemcpyHostToDevice);
    cudaFreeHost(h_data);
}

// void init_vector(__half* d_data, int len) {
//     // a rough piece of code considering no performance 
//     __half* h_data;
//     cudaMallocHost((void**) &h_data, sizeof(__half) * len);
//     for (size_t i = 0; i < len; ++ i) {
//         h_data[i] = __float2half(1);
//     }
//     cudaMemcpy(d_data, h_data, sizeof(__half) * len, cudaMemcpyHostToDevice);
//     cudaFreeHost(h_data);
// }

void init_vector(char* d_data, int len) {
    float* f_data;
    cudaMallocHost((void**) &f_data, sizeof(float) * len);
    for (size_t i = 0; i < len; ++ i) {
        //f_data[i] = i + 1;
        f_data[i] = (float)(i + 1);
    }
    char* h_data;
    cudaMallocHost((void**) &h_data, sizeof(char) * len);
    for (size_t i = 0; i < len; ++ i) {
        h_data[i] = (char) round(f_data[i] * 128);
    }

    cudaMemcpy(d_data, h_data, sizeof(char) * cudaMemcpyHostToDevice, cudaMemcpyHostToDevice);
    cudaFreeHost(h_data);
    cudaFreeHost(f_data);
}

// void print_matrix(const int* d_mat, int m, int n) {
//     // a rough piece of code considering no performance 
//     int* h_mat;
//     cudaMallocHost((void**) &h_mat, sizeof(int) * m * n);
//     cudaMemcpy(h_mat, d_mat, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
//     for (size_t irow = 0; irow < min(m, 10); ++ irow) {
//         for (size_t icol = 0; icol < min(n, 10); ++ icol) {
//             std::cout << h_mat[n*irow + icol] << " " ;
//         }
//         std::cout << std::endl;
//     }
//     cudaFreeHost(h_mat);
// }

void print_C(const int* d_mat, int m, int n) {
    // a rough piece of code considering no performance 
    int* h_mat;
    cudaMallocHost((void**) &h_mat, sizeof(int) * m * n);
    cudaMemcpy(h_mat, d_mat, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    for (size_t irow = 0; irow < min(m, 10); ++ irow) {
        for (size_t icol = 0; icol < min(n, 10); ++ icol) {
            //std::cout << ((float)h_mat[n*irow + icol]) / (128 * 128) << " " ;
            std::cout << h_mat[n*irow + icol] / (128 * 128) << " " ;
        }
        std::cout << std::endl;
    }
    cudaFreeHost(h_mat);
}

// void print_matrix(const __half* d_mat, int m, int n) {
//     // a rough piece of code considering no performance 
//     __half* h_mat;
//     cudaMallocHost((void**) &h_mat, sizeof(__half) * m * n);
//     cudaMemcpy(h_mat, d_mat, sizeof(__half) * m * n, cudaMemcpyDeviceToHost);
//     for (size_t irow = 0; irow < min(m, 10); ++ irow) {
//         for (size_t icol = 0; icol < min(n, 10); ++ icol) {
//             std::cout << __half2float(h_mat[n*irow + icol]) << " " ;
//         }
//         std::cout << std::endl;
//     }
//     cudaFreeHost(h_mat);
// }

void warmUp ()
{
  const int N = 1000;

  // init stream and cublas
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle); // takes hundrads of milliseconds for first time

  // cudaMemcpy
  float* h_data(NULL);
  cudaMallocHost(&h_data, sizeof(float) * N);
  for (int i = 0; i < N; ++ i) h_data[i] = 1.0f * i;
  float* d_data(NULL);
  cudaMalloc((void**)&d_data, sizeof(float) * N);
  cudaMemcpyAsync(d_data, h_data, sizeof(float) * N, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(h_data, d_data, sizeof(float) * N, cudaMemcpyDeviceToHost, stream1);
  for (int i = 0; i < N; ++ i) assert(h_data[i] == 1.0f * i);
  cudaStreamSynchronize(stream1);
  cudaFreeHost(h_data);
  cudaFree(d_data);
}

int run(int m, int k, int n, uint64_t niters) {
    cout <<"m,k,n: " << m << ", " << k << "," << n << endl;
    cout << "datatype: " << DATA_TYPE << endl;

    datatypeA *d_A;
    datatypeB *d_B;
    datatypeC *d_C;
    cudaMalloc((void**) &d_A, sizeof(datatypeA) * m * k);
    cudaMalloc((void**) &d_B, sizeof(datatypeB) * k * n);
    cudaMalloc((void**) &d_C, sizeof(datatypeC) * m * n);
    cudaMemset(d_C, 0, sizeof(datatypeC) * m * n);
    init_vector(d_A, m * k);
    init_vector(d_B, k * n);
    init_vector(d_A, m * n);

#if LOG_LEVEL >= 2
    cout << "A:" << endl;
    //print_matrix(d_A, m, k);
    cout << "B:" << endl;
    //print_matrix(d_B, k, n);
#endif
    
    cublasHandle_t handle;
    cublasCreate(&handle);
#if DATA_TYPE == 2
    int alpha = 1;
    int beta = 0;
    // cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    // cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO5;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    cout << "algo:" << algo << endl;
#else
    cout << "not implemented" << endl;
    exit(0);
#endif
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    int lda = (transA == CUBLAS_OP_N) ? m : k;
    int ldb = (transB == CUBLAS_OP_N) ? k : n;
    int ldc = n;

    cudaError_t ret;
    cudaEvent_t events[2];

    for (auto & event : events) {
      ret = cudaEventCreate(&event);
      if (ret != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
      }
    }
    
    // Record an event at the start of a series of GEMMs
    ret = cudaEventRecord(events[0]);
    if (ret != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }

    for (int iter = 0; iter < niters; ++iter) {
        #if DATA_TYPE == 0 // float
            cout << "not implemented" << endl;
            exit(0);
        #elif DATA_TYPE == 1 // half
            cout << "not implemented" << endl;
            exit(0);
        #elif DATA_TYPE == 2
            cublasGemmEx(handle, transA, transB, (int)m, (int)n, (int)k, 
                &alpha, d_A, CUDA_R_8I, (int)lda, d_B, CUDA_R_8I, (int)ldb, &beta, 
                d_C, CUDA_R_32I, (int)ldc, CUDA_R_32I, algo);    
        #endif
      }

    ret = cudaEventRecord(events[1]);
    if (ret != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(ret) << std::endl;
    return -1;
    }      
       
    // Wait for work on the device to complete.
    ret = cudaEventSynchronize(events[1]);
    if (ret != cudaSuccess) {
        std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }

    float runtime_ms = 0;
    ret = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (ret != cudaSuccess) {
        std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }
    runtime_ms = double(runtime_ms) / double(niters);

    // Cleanup
    for (auto event : events) {
        (void)cudaEventDestroy(event);
    }

#if LOG_LEVEL >= 2
    cout << "C:" << endl;
    print_C(d_C, m, n);
#endif

    cout << "Runtime: " << runtime_ms << " ms" << endl;
    cout << "total time: " << runtime_ms * niters << " ms for " << niters << " loops" << endl;
    cout << "transA, transB: " << ((transA == CUBLAS_OP_N) ? "N" : "T") << " ,"
         << ((transB == CUBLAS_OP_N) ? "N" : "T") << endl;

    // release
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

int main(int argc, const char **argv) {
    #if CHECK_ARCH
        if (NVCC_ARCHS ==0) {
            std::cerr << "NVCC_ARCHS must specified" << std::endl;
            // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
            return 0;
        }

        // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
        //
        // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
        if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
        std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;
        return 0;
        }

        cudaDeviceProp props;

        cudaError_t error = cudaGetDeviceProperties(&props, 0);
        if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
        }
    
        if (props.major != 7) {
        std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
                    << std::endl;
        // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
        return 0;
        }
    #endif

    cutlass::CommandLine cmd(argc, argv);
    int m = 2, k = 3, n = 2;
    uint64_t niters = 1;
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("k", k);        
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("niters", niters);

    warmUp();

    run(m, k, n, niters);

    return 0;
  }