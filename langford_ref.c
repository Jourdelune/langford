/* langford_ref.c -- reference CPU implementation of Godfrey's algebraic method
 *
 * Langford pairing L(2,n): place 1,1,2,2,...,n,n in 2n slots so the two copies
 * of v are exactly v+1 apart (v slots between them).
 *
 * Polynomial model (Godfrey 2002):
 *   F(n,X) = prod_{i=2}^{n+1} ( sum_{k=1}^{2n-i} x_k x_{k+i} )
 * The coefficient of x_1 x_2 ... x_{2n} in F is V(n) = 2*L(2,n)
 * (reflections counted as distinct).  Extracted by +-1 evaluation:
 *   V(n) = 2^{-2n} * sum_{X in {-1,1}^{2n}} (prod_k x_k) F(n,X)
 *
 * Symmetry group {id, neg-all, neg-even, neg-odd} = Z2xZ2 acts simply
 * transitively on (x_{2n-1}, x_{2n}) and leaves (prod x)F invariant for every
 * valid Langford order (n = 0 or 3 mod 4).  So we fix x_{2n-1}=x_{2n}=+1 and
 * multiply by 4.
 *
 * This file recomputes every P_i from scratch with popcount: slow but as
 * close to the definition as possible.  It is the ground truth the GPU kernel
 * is validated against.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static void print_i128(__int128 v){
    if (v < 0) { putchar('-'); v = -v; }
    char buf[48]; int p = 0;
    if (!v) buf[p++] = '0';
    while (v) { buf[p++] = '0' + (int)(v % 10); v /= 10; }
    while (p) putchar(buf[--p]);
}

int main(int argc, char **argv){
    if (argc < 2) { fprintf(stderr, "usage: %s n\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    int M = 2*n;                 /* number of positions / variables      */
    if (n < 3 || M > 40) { fprintf(stderr, "n out of range for reference\n"); return 1; }
    if (n % 4 != 0 && n % 4 != 3)
        fprintf(stderr, "warning: n=%d has no Langford pairing (expect 0)\n", n);

    int L = M - 2;               /* free bits: 0..M-3                    */
    uint64_t total = 1ULL << L;
    uint64_t mask_i[64];
    for (int i = 2; i <= n+1; i++) mask_i[i] = (M-i >= 64) ? ~0ULL : ((1ULL << (M-i)) - 1);

    __int128 S = 0;
    uint64_t b = 0;              /* b_k = 1 means x_{k+1} = -1           */

    for (uint64_t t = 0; ; t++) {
        /* evaluate (prod x) * F at the current sign vector */
        __int128 prod = 1;
        for (int i = 2; i <= n+1; i++) {
            int q = __builtin_popcountll((b ^ (b >> i)) & mask_i[i]);
            prod *= (__int128)((M - i) - 2*q);
            if (!prod) break;
        }
        if (__builtin_popcountll(b) & 1) S -= prod; else S += prod;

        if (t + 1 == total) break;
        b ^= 1ULL << __builtin_ctzll(t + 1);   /* binary reflected Gray code */
    }

    S *= 4;                       /* Z2xZ2 symmetry                      */
    __int128 V = S >> M;          /* divide by 2^{2n}                    */
    if (S & (((__int128)1 << M) - 1)) { fprintf(stderr, "INTERNAL ERROR: sum not divisible by 2^%d\n", M); return 2; }

    printf("n=%2d  V(n)=2*L(2,n) = ", n); print_i128(V);
    printf("   L(2,n) = "); print_i128(V/2); printf("\n");
    return 0;
}
