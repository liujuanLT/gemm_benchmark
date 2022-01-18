#nvcc -gencode arch=compute70,code=sm_70 gemm_cublas.cu -lcublas -std=c++11 -o example.exe 
nvcc -gencode=arch=compute_86,code=sm_86 -O3 gemm_cublas_ex.cu -lcublas -std=c++11 -o example.exe 