# ref: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

# Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A2000, A3000, RTX A4000, A5000, A6000, NVIDIA A40, GA106 – RTX 3060, GA104 – RTX 3070, GA107 – RTX 3050, RTX A10, RTX A16, RTX A40, A2 Tensor Core GPU
nvcc -gencode=arch=compute_86,code=sm_86 -O3 gemm_cublas_ex.cu -lcublas -std=c++11 -o example.exe 

#nvcc -gencode arch=compute70,code=sm_70 gemm_cublas.cu -lcublas -std=c++11 -o example.exe 