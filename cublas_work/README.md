``` shell
nvcc  gemm_cublas.cu -lcublas -std=c++11 -o example.exe
./example.exe  --m=10 --k=10 --n=10 --niters=1000

``` shell


# different types of transpose type
```
m,k,n: 100, 100,100
Runtime: 0.00552858 ms
total time: 5.52858 ms for 1000 loops
transA, transB: N ,N

m,k,n: 100, 100,100
Runtime: 0.00567194 ms
total time: 5.67194 ms for 1000 loops
transA, transB: T ,N

m,k,n: 100, 100,100
Runtime: 0.00563507 ms
total time: 5.63507 ms for 1000 loops
transA, transB: N ,T

m,k,n: 100, 100,100
Runtime: 0.00545792 ms
total time: 5.45792 ms for 1000 loops
transA, transB: T ,T
```