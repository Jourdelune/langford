/* =====================================================================
 * langford.cu -- single-GPU implementation of Godfrey's algebraic method
 *                for counting Langford pairings L(2,n).
 *
 *   V(n) = 2*L(2,n) = 2^{-2n} * SUM_{X in {-1,1}^{2n}} (prod_k x_k) * F(n,X)
 *   F(n,X) = prod_{i=2}^{n+1} ( sum_{k=1}^{2n-i} x_k x_{k+i} )
 *
 * Implementation notes (all of this is what makes it fast):
 *
 *  1. Sign vector held as a bitmask b (b_k = 1  <=>  x_{k+1} = -1).
 *     P_i = (2n-i) - 2*popcount( (b ^ (b>>i)) & mask_i )    (autocorrelation)
 *
 *  2. Binary reflected Gray code over the free bits: one bit flips per step,
 *     so each P_i moves by -4, 0 or +4.  The flipped bit index is warp-uniform,
 *     which makes the per-step constant tables a broadcast from __constant__.
 *
 *  3. The n values P_i are kept as 31/32 *bytes* packed in 8 uint32 registers,
 *     biased by +128.  The whole delta vector is produced with a bit->byte
 *     expansion trick,  expand(v) = ((v & 0xF) * 0x00810204) & 0x04040404,
 *     which turns 4 mask bits into 4 bytes of value 4 in three instructions.
 *     Range analysis guarantees no carry ever crosses a byte lane:
 *        P_i in [-62,62] -> byte in [66,190]; +8 -> <=198; -4 -> >=62.
 *
 *  4. b and its bit reversal rb are both maintained, so the two operands of
 *     the delta (x_{j+i} and x_{j-i}) are single shifts instead of gathers.
 *
 *  5. The product of the n factors is a *balanced tree*, not a running
 *     product: 6-bit x 6-bit -> 12 -> 24 -> 48 -> 96 -> 160 bits.  That is
 *     ~45 word-multiplies instead of ~85 for the sequential version.
 *
 *  6. Everything is exact modulo 2^160.  The true sum is 2^{2n}*V(n) with
 *     V(31) ~ 2^84, i.e. ~2^146 < 2^160, so no CRT and no primes are needed:
 *     truncated 160-bit arithmetic *is* the exact answer.  The low 2n bits of
 *     the sum must come out zero, which is a free end-to-end self-check.
 *
 *  7. Symmetry group {id, negate-all, negate-even, negate-odd} = Z2 x Z2 acts
 *     simply transitively on (x_{2n-1}, x_{2n}) and fixes (prod x)F for every
 *     valid Langford order.  Both bits are pinned to +1 and the sum is
 *     multiplied by 4, so only 2^{2n-2} vectors are enumerated.
 * ===================================================================== */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d: %s\n",#x,__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(1);} }while(0)

/* ---- per-flipped-bit constant tables (index j = flipped bit) ---- */
__constant__ uint32_t cValidA[32];
__constant__ uint32_t cValidB[32];
__constant__ uint32_t cSub[32][8];   /* byte-wise 2*validA + 2*validB */
__constant__ int      cShiftB[32];

/* ---------------- 160-bit helpers (5 x uint32 limbs) ---------------- */
#define ADD160(z,x) asm volatile( \
    "add.cc.u32  %0, %0, %5;\n\t addc.cc.u32 %1, %1, %6;\n\t"  \
    "addc.cc.u32 %2, %2, %7;\n\t addc.cc.u32 %3, %3, %8;\n\t"  \
    "addc.u32    %4, %4, %9;"                                   \
    : "+r"(z[0]),"+r"(z[1]),"+r"(z[2]),"+r"(z[3]),"+r"(z[4])    \
    : "r"(x[0]),"r"(x[1]),"r"(x[2]),"r"(x[3]),"r"(x[4]))

#define SUB160(z,x) asm volatile( \
    "sub.cc.u32  %0, %0, %5;\n\t subc.cc.u32 %1, %1, %6;\n\t"  \
    "subc.cc.u32 %2, %2, %7;\n\t subc.cc.u32 %3, %3, %8;\n\t"  \
    "subc.u32    %4, %4, %9;"                                   \
    : "+r"(z[0]),"+r"(z[1]),"+r"(z[2]),"+r"(z[3]),"+r"(z[4])    \
    : "r"(x[0]),"r"(x[1]),"r"(x[2]),"r"(x[3]),"r"(x[4]))

/* ============================ kernel ============================ */
template<int N>
__global__ __launch_bounds__(256)
void godfrey(uint64_t base, int innerBits, uint32_t * __restrict__ out)
{
    const int M = 2*N;                       /* number of positions */
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t b  = base | (tid << innerBits);
    uint64_t rb = __brevll(b) >> (64 - M);   /* rb_p = b_{M-1-p} */

    /* ---- initialise the packed P vector (bias +128) ---- */
    uint32_t r[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) r[k] = 0x81818181u;   /* unused lanes = +1 */
    #pragma unroll
    for (int i = 2; i <= N+1; i++) {
        const int w = M - i;
        const uint64_t mk = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
        int P = w - 2*__popcll((b ^ (b >> i)) & mk);
        const int l = i - 2;
        r[l >> 2] = (r[l >> 2] & ~(0xFFu << (8*(l & 3))))
                  | ((uint32_t)(P + 128) << (8*(l & 3)));
    }

    int sign = (__popcll(b) & 1) ? -1 : 1;
    uint32_t acc[5] = {0,0,0,0,0};

    const uint64_t steps = 1ULL << innerBits;
    for (uint64_t t = 0; ; t++) {

        /* ---------- evaluate (prod x) * prod_i P_i ---------- */
        int p[16];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            uint32_t d = r[k] ^ 0x80808080u;          /* -> two's complement bytes */
            int v0 = (int)(signed char)(d);
            int v1 = (int)(signed char)(d >> 8);
            int v2 = (int)(signed char)(d >> 16);
            int v3 = (int)(signed char)(d >> 24);
            p[2*k]     = v0 * v1;                    /* |.| <= 3844  */
            p[2*k + 1] = v2 * v3;
        }
        int q[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) q[k] = p[2*k] * p[2*k+1];      /* <= 1.5e7 */

        long long s[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) s[k] = (long long)q[2*k] * q[2*k+1]; /* < 2^48 */

        int neg = (sign < 0);
        unsigned long long u[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            neg ^= (s[k] < 0);
            u[k] = (unsigned long long)(s[k] < 0 ? -s[k] : s[k]);
        }

        /* level 4 : two 96-bit products (3 limbs each) */
        uint32_t A[3], C[3];
        {
            uint32_t a0=(uint32_t)u[0], a1=(uint32_t)(u[0]>>32);
            uint32_t c0=(uint32_t)u[1], c1=(uint32_t)(u[1]>>32);
            uint64_t p00=(uint64_t)a0*c0, p01=(uint64_t)a0*c1,
                     p10=(uint64_t)a1*c0, p11=(uint64_t)a1*c1;
            uint64_t m = (p00>>32) + (uint32_t)p01 + (uint32_t)p10;
            A[0]=(uint32_t)p00;
            A[1]=(uint32_t)m;
            A[2]=(uint32_t)((m>>32) + (p01>>32) + (p10>>32) + p11);
        }
        {
            uint32_t a0=(uint32_t)u[2], a1=(uint32_t)(u[2]>>32);
            uint32_t c0=(uint32_t)u[3], c1=(uint32_t)(u[3]>>32);
            uint64_t p00=(uint64_t)a0*c0, p01=(uint64_t)a0*c1,
                     p10=(uint64_t)a1*c0, p11=(uint64_t)a1*c1;
            uint64_t m = (p00>>32) + (uint32_t)p01 + (uint32_t)p10;
            C[0]=(uint32_t)p00;
            C[1]=(uint32_t)m;
            C[2]=(uint32_t)((m>>32) + (p01>>32) + (p10>>32) + p11);
        }

        /* level 5 : 96 x 96 -> keep 160 bits (column-wise, exact) */
        uint32_t z[5];
        {
            uint64_t P00=(uint64_t)A[0]*C[0];
            uint64_t P01=(uint64_t)A[0]*C[1], P10=(uint64_t)A[1]*C[0];
            uint64_t P02=(uint64_t)A[0]*C[2], P11=(uint64_t)A[1]*C[1], P20=(uint64_t)A[2]*C[0];
            uint64_t P12=(uint64_t)A[1]*C[2], P21=(uint64_t)A[2]*C[1];
            uint64_t P22=(uint64_t)A[2]*C[2];
            uint64_t carry, s0;
            s0 = (uint32_t)P00;                              z[0]=(uint32_t)s0;
            carry = (s0>>32) + (P00>>32);
            s0 = carry + (uint32_t)P01 + (uint32_t)P10;      z[1]=(uint32_t)s0;
            carry = (s0>>32) + (P01>>32) + (P10>>32);
            s0 = carry + (uint32_t)P02 + (uint32_t)P11 + (uint32_t)P20; z[2]=(uint32_t)s0;
            carry = (s0>>32) + (P02>>32) + (P11>>32) + (P20>>32);
            s0 = carry + (uint32_t)P12 + (uint32_t)P21;      z[3]=(uint32_t)s0;
            carry = (s0>>32) + (P12>>32) + (P21>>32);
            s0 = carry + (uint32_t)P22;                      z[4]=(uint32_t)s0;
        }

        if (neg) { SUB160(acc, z); } else { ADD160(acc, z); }

        if (t + 1 == steps) break;

        /* ---------------- Gray-code step ---------------- */
        const int j = __ffsll((long long)(t + 1)) - 1;   /* warp-uniform */
        const uint32_t cm = -(uint32_t)((b >> j) & 1ULL);
        uint32_t Am = ((uint32_t)(b  >> (j + 2))       ^ cm) & cValidA[j];
        uint32_t Bm = ((uint32_t)(rb >> cShiftB[j])    ^ cm) & cValidB[j];

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            uint32_t ta = (((Am >> (4*k)) & 0xFu) * 0x00810204u) & 0x04040404u;
            uint32_t tb = (((Bm >> (4*k)) & 0xFu) * 0x00810204u) & 0x04040404u;
            r[k] = r[k] + ta + tb - cSub[j][k];
        }

        b    ^= 1ULL << j;
        rb   ^= 1ULL << (M - 1 - j);
        sign  = -sign;
    }

    /* ---------------- block reduction ---------------- */
    __shared__ uint32_t sm[256][5];
    #pragma unroll
    for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = acc[k];
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint32_t x[5], y[5];
            #pragma unroll
            for (int k = 0; k < 5; k++) { x[k]=sm[threadIdx.x][k]; y[k]=sm[threadIdx.x+stride][k]; }
            ADD160(x, y);
            #pragma unroll
            for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = x[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        #pragma unroll
        for (int k = 0; k < 5; k++) out[blockIdx.x*5 + k] = sm[0][k];
}

