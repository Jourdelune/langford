#!/bin/sh
set -e
nvcc -O3 -arch=sm_89                   -o langford6 langford6.cu   # reference
nvcc -O3 -arch=sm_89 -maxrregcount 32  -o langford5 langford5.cu
nvcc -O3 -arch=sm_89                   -o langford4 langford4.cu
nvcc -O3 -arch=sm_89                   -o langford3 langford3.cu
gcc  -O2 -o oe_check oe_check.c ; gcc -O2 -o oe_ref oe_ref.c ; gcc -O2 -o oe_ref2 oe_ref2.c
gcc  -O2 -o pfaff_test pfaff_test.c ; gcc -O2 -o pfaff_zk pfaff_zk.c
echo "ok"
