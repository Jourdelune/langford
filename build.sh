#!/bin/sh
# Compilation portable : la machine de calcul est souvent louee (vast.ai), donc
# l'architecture n'est pas connue a l'avance.
#   1. -arch=native si le toolkit le supporte (CUDA >= 11.5)
#   2. sinon on lit la compute capability et on la traduit en sm_XX
#   3. si le toolkit est trop vieux pour la carte (typiquement CUDA < 12.8 pour
#      Blackwell sm_120), on le dit clairement au lieu de produire un binaire
#      qui echouera au lancement.
set -e
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
if nvcc -arch=native -x cu -c /dev/null -o /dev/null 2>/dev/null; then
  ARCH="-arch=native"
elif [ -n "$CC" ] && nvcc -arch=sm_$CC -x cu -c /dev/null -o /dev/null 2>/dev/null; then
  ARCH="-arch=sm_$CC"
else
  echo "ERREUR : le toolkit CUDA installe ne connait pas sm_${CC:-?}." >&2
  nvcc --version | tail -2 >&2
  echo "  Blackwell (RTX 50xx, sm_120) exige CUDA >= 12.8." >&2
  echo "  Sur vast.ai : choisir une image nvidia/cuda:12.8+-devel." >&2
  exit 1
fi
echo "architecture : $ARCH"
nvcc -O3 $ARCH                   -o langford6 langford6.cu    # reference
nvcc -O3 $ARCH -maxrregcount 32  -o langford5 langford5.cu 2>/dev/null || true
nvcc -O3 $ARCH                   -o langford4 langford4.cu 2>/dev/null || true
nvcc -O3 $ARCH                   -o langford3 langford3.cu 2>/dev/null || true
for f in oe_check oe_ref oe_ref2 pfaff_test pfaff_zk fiber zfrac single powersums framework; do
  [ -f $f.c ] && gcc -O2 -o $f $f.c -lm 2>/dev/null || true
done
[ -f dpstates.cpp ] && g++ -O2 -o dpstates dpstates.cpp -lm 2>/dev/null || true
[ -f struct.c ] && gcc -O2 -o struct struct.c -lm 2>/dev/null || true
[ -f toeplitz.c ] && gcc -O2 -o toeplitz toeplitz.c 2>/dev/null || true
[ -f partial.c ] && gcc -O2 -o partial partial.c 2>/dev/null || true
[ -f find_langford.c ] && gcc -O2 -o find_langford find_langford.c 2>/dev/null || true
[ -f estimate.c ] && gcc -O2 -fopenmp -o estimate estimate.c 2>/dev/null || true
echo ok
