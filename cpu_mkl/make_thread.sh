# /opt/intel/oneapi/compiler/2022.0.1/linux/bin/intel64/icc  -qmkl dgemm_thread_effect.c -o thread_dgemm
# /opt/intel/oneapi/compiler/2022.0.1/linux/bin/intel64/icc  -qmkl dgemm_with_timing.c -o timing_dgemm
/opt/intel/oneapi/compiler/2022.0.1/linux/bin/intel64/icc  -qmkl gemm_dnnl.cpp -o example -I /opt/intel/oneapi/dnnl/latest/cpu_iomp/include -L /opt/intel/oneapi/dnnl/latest/cpu_iomp/lib -ldnnl