// ref: https://github.com/mnicely/cublasLt_examples/blob/master/cublasLt_INT8_TCs.cu

/* Includes, system */
#include <cstdio>

/* Includes, cuda & thrust*/
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
using namespace std;

#if 0
// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************
#endif

#define CUDA_RT_CALL(call) call 

#ifndef CHECK_RESULT
#define CHECK_RESULT 0
#endif

#ifndef DATA_TYPE
#define DATA_TYPE 2 // 0: float, 1: half, 2:int8-int32, 3:int8-fp32
#endif

#if DATA_TYPE == 2
    typedef int8_t  dataTypeI;
    typedef int32_t dataTypeO;
    typedef int32_t dataTypeS;
    auto constexpr cudaDataTypeI   = CUDA_R_8I;
    auto constexpr cudaDataTypeO   = CUDA_R_32I;
    auto constexpr cudaComputeType = CUBLAS_COMPUTE_32I; // here I fixed, otherwise it raises error
#elif DATA_TYPE == 3
    typedef int8_t  dataTypeI;
    typedef float dataTypeO; // TODO, check
    typedef float dataTypeS; // TODO, check
    auto constexpr cudaDataTypeI   = CUDA_R_8I;
    auto constexpr cudaDataTypeO   = CUDA_R_32F;
    auto constexpr cudaComputeType = CUBLAS_COMPUTE_32F; // here I fixed, otherwise it raises error
#endif

auto constexpr maxN            = 2048;

int roundoff( int v, int d ) {
    return ( v + d - 1 ) / d * d;
}

void LtIgemmTensor( cublasLtHandle_t ltHandle,
                    int const &      m,
                    int const &      n,
                    int const &      k,
                    dataTypeI const *A,
                    int const &      lda,
                    dataTypeI const *B,
                    int const &      ldb,
                    dataTypeO *      C,
                    int const &      ldc ) {

    cublasLtMatmulDesc_t   matmulDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    dataTypeS         alpha = 1, beta = 0;
    cublasLtOrder_t   rowOrder    = CUBLASLT_ORDER_ROW;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // The tensor operations IGEMM kernels require specialized memory order of data.
    cublasLtMatrixTransformDesc_t transformDesc = nullptr;
    dataTypeI *                   Atransform = nullptr, *Btransform = nullptr;
    dataTypeO *                   Ctransform     = nullptr;
    cublasLtMatrixLayout_t        AtransformDesc = nullptr, BtransformDesc = nullptr, CtransformDesc = nullptr;

    float const     transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int const ldatransform = 32 * m;
    int const ldbtransform = 32 * roundoff( n, 8 );
    int const ldctransform = 32 * m;

    CUDA_RT_CALL( cudaMalloc( &Atransform, sizeof( dataTypeI ) * roundoff( k, 32 ) / 32 * ldatransform ) );
    CUDA_RT_CALL( cudaMalloc( &Btransform, sizeof( dataTypeI ) * roundoff( k, 32 ) / 32 * ldbtransform ) );
    CUDA_RT_CALL( cudaMalloc( &Ctransform, sizeof( dataTypeO ) * roundoff( n, 32 ) / 32 * ldctransform ) );

    CUDA_RT_CALL( cublasLtMatrixTransformDescCreate( &transformDesc, CUDA_R_32F ) );
    CUDA_RT_CALL( cublasLtMatmulDescCreate( &matmulDesc, cudaComputeType, cudaDataTypeO ) );

    // Tensor operations IGEMM kernels only support NT gemm
    CUDA_RT_CALL( cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof( opTranspose ) ) );

    // --------------------------------------
    // Create descriptors for the original matrices
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &Adesc, cudaDataTypeI, m, k, lda ) );

    // B matrix is non-transposed, but transposed matrix is needed -
    // describe matrix as row major.
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &Bdesc, cudaDataTypeI, n, k, ldb ) );
    CUDA_RT_CALL(
        cublasLtMatrixLayoutSetAttribute( Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );

    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &Cdesc, cudaDataTypeO, m, n, ldc ) );

    // -----------------------------------------------------------
    // Create descriptors for the transformed matrices
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &AtransformDesc, cudaDataTypeI, m, k, ldatransform ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof( order_COL32 ) ) );

    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &BtransformDesc, cudaDataTypeI, n, k, ldbtransform ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof( order_COL4_4R2_8C ) ) );

    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &CtransformDesc, cudaDataTypeO, m, n, ldctransform ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof( order_COL32 ) ) );

    // --------------------------------------------------------
    // Transforms and computation
    CUDA_RT_CALL( cublasLtMatrixTransform( ltHandle,
                                           transformDesc,
                                           &transformAlpha,
                                           A,
                                           Adesc,
                                           &transformBeta,
                                           nullptr,
                                           nullptr,
                                           Atransform,
                                           AtransformDesc,
                                           0 ) );

    CUDA_RT_CALL( cublasLtMatrixTransform( ltHandle,
                                           transformDesc,
                                           &transformAlpha,
                                           B,
                                           Bdesc,
                                           &transformBeta,
                                           nullptr,
                                           nullptr,
                                           Btransform,
                                           BtransformDesc,
                                           0 ) );

    // No need to transform C matrix as beta is assumed to be 0
    CUDA_RT_CALL( cublasLtMatmul( ltHandle,
                                  matmulDesc,
                                  &alpha,
                                  Atransform,
                                  AtransformDesc,
                                  Btransform,
                                  BtransformDesc,
                                  &beta,
                                  Ctransform,
                                  CtransformDesc,
                                  Ctransform,
                                  CtransformDesc,
                                  nullptr,
                                  nullptr,
                                  0,
                                  0 ) );

    // Transform the outputs to COL order
    CUDA_RT_CALL( cublasLtMatrixTransform( ltHandle,
                                           transformDesc,
                                           &transformAlpha,
                                           Ctransform,
                                           CtransformDesc,
                                           &transformBeta,
                                           nullptr,
                                           nullptr,
                                           C,
                                           Cdesc,
                                           0 ) );

    // Descriptors are no longer needed as all GPU work was already
    // enqueued.
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( CtransformDesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( BtransformDesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( AtransformDesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Cdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Bdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Adesc ) );
    CUDA_RT_CALL( cublasLtMatmulDescDestroy( matmulDesc ) );
    CUDA_RT_CALL( cublasLtMatrixTransformDescDestroy( transformDesc ) );

    // Wait until device is done before freeing transformed buffers
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    CUDA_RT_CALL( cudaFree( Ctransform ) );
    CUDA_RT_CALL( cudaFree( Btransform ) );
    CUDA_RT_CALL( cudaFree( Atransform ) );
}

/* Host implementation of a simple version of IGEMM */
static void simple_igemm( int const &      m,
                          int const &      k,
                          int const &      n,
                          dataTypeS const &alpha,
                          dataTypeI const *A,
                          dataTypeI const *B,
                          dataTypeS const &beta,
                          dataTypeO *      C ) {

    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < m; ++j ) {
            dataTypeO prod = 0;

            for ( int k = 0; k < n; ++k ) {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

/* Main */
void calculate( int const &m, int const &n, int const &k ) {

    dataTypeS alpha = 1, beta = 0;
    int       lda = m, ldb = k, ldc = m;

    size_t sizeA = m * k;
    size_t sizeB = k * n;
    size_t sizeC = m * n;

    dataTypeO error_norm = 0;
    dataTypeO ref_norm   = 0;
    dataTypeO diff       = 0;

    cublasLtHandle_t handle;

    /* Initialize cuBLASLt */
    CUDA_RT_CALL( cublasLtCreate( &handle ) );

    printf( "cublasLt %dx%dx%d test running..\n", m, n, k );

    /* Allocate host memory for the matrices */
    thrust::host_vector<dataTypeI> h_A( sizeA, 0 );
    thrust::host_vector<dataTypeI> h_B( sizeB, 0 );
    thrust::host_vector<dataTypeO> h_C( sizeC, 0 );
    thrust::host_vector<dataTypeO> h_C_ref( sizeC, 0 );

    /* Fill the matrices with test data */
    /* Assume square matrices */
    for ( int i = 0; i < m * m; i++ ) {
        h_A[i] = rand( ) / static_cast<int8_t>( RAND_MAX );
        ;
        h_B[i] = rand( ) / static_cast<int8_t>( RAND_MAX );
        ;
    }

    /* Allocate device memory for the matrices */
    thrust::device_vector<dataTypeI> d_A( h_A );
    thrust::device_vector<dataTypeI> d_B( h_B );
    thrust::device_vector<dataTypeO> d_C( sizeC, 0 );

    /* Retrieve raw pointer for device data */
    dataTypeI *d_A_ptr = thrust::raw_pointer_cast( &d_A[0] );
    dataTypeI *d_B_ptr = thrust::raw_pointer_cast( &d_B[0] );
    dataTypeO *d_C_ptr = thrust::raw_pointer_cast( &d_C[0] );

    /* Performs operation using plain C code */
    simple_igemm( m, n, k, alpha, h_A.data( ), h_B.data( ), beta, h_C_ref.data( ) );

    /* cublasLt with int8/TCs */
    LtIgemmTensor( handle, m, n, k, d_A_ptr, lda, d_B_ptr, ldb, d_C_ptr, ldc );

    /* Allocate host memory for reading back the result from device memory */
    h_C = d_C;

    /* Check result against reference */
    for ( int i = 0; i < m * m; ++i ) {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = static_cast<dataTypeO>( sqrt( static_cast<double>( error_norm ) ) );
    ref_norm   = static_cast<dataTypeO>( sqrt( static_cast<double>( ref_norm ) ) );

#if CHECK_RESULT != 0
    if ( fabs( ref_norm ) < 1e-7 )
        throw std::runtime_error( "!!!! reference norm is 0\n" );

    /* Shutdown */
    CUDA_RT_CALL( cublasLtDestroy( handle ) );

    if ( error_norm / ref_norm < 1e-4f )
        printf( "cuBLASLt IGEMM test passed.\n" );
    else
        throw std::runtime_error( "!!!! cuBLASLt IGEMM test failed.\n" );
#endif
}

/* Main */
int main( int argc, char **argv ) {

    int dev {};
    CUDA_RT_CALL( cudaGetDevice( &dev ) );

    // Ensure GPU found is compute capability 7.2 or greater
    cudaDeviceProp deviceProp;
    CUDA_RT_CALL( cudaGetDeviceProperties( &deviceProp, dev ) );

    if ( deviceProp.major > 6 ) {
        if ( deviceProp.minor < 2 ) {
            throw std::runtime_error( "ERROR: This sample utilizes compute capability 7.2 or greater!" );
        }
    } else {
        throw std::runtime_error( "ERROR: This sample utilizes compute capability 7.2 or greater!" );
    }

    // warm up
    // for (int i = 0; i < 5; ++ i)
    //     calculate( 64, 64, 64); // m, n, k

    // // Compute square matrices
    // for ( int i = 1; i <= maxN; i *= 2 )
    //     calculate( i, 256, 64 ); // m, n, k
    for (int i = 0; i < 10; ++ i)
    calculate( 64, 256, 64 ); // m, n, k

    return ( EXIT_SUCCESS );
}