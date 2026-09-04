/* =====================================================================
 * langford5.cu -- v4 + half-state hot loop + constant-index Gray steps.
 *
 * New in v4.  Measured on a 4070: the 160-bit product tree is 70.6% of the
 * runtime (37.2 Gsums/s with it, 126.5 without) and 87.1% of those products
 * are identically zero: A_i(b) = 0 is possible exactly for even i (A_i has
 * M-i terms of +-1, so A_i = M-i mod 2), and it happens for at least one of
 * the 16 even lags in 87.1% of the vectors at n=31.
 *
 * v4 detects that in ~15 integer ops and computes the product only for the
 * survivors.  A per-element branch is useless (P(all 32 lanes dead) = 0.87^32
 * = 1.2%), so survivors are compacted into a per-warp ring buffer in shared
 * memory and drained 32 at a time -- every product is then computed by a
 * FULL warp.
 *
 * Two changes make the test cheap:
 *   - SWAR lane re-layout: the 16 even lags (the only ones that can vanish)
 *     are packed into words 0-3, so the test touches 4 words, not 8.  The
 *     Gray-code update keeps the same instruction count with a stride-2
 *     expander:  ((v>>s) & 0x55) * 0x104104 & 0x04040404.
 *   - the term's sign rides in the unused lane 31 as +-1 (0x81 / 0x7F), so a
 *     survivor is 8 words and needs no separate sign in the queue.
 *
 * ===================================================================== */
/* =====================================================================
 * (v3 header, still accurate for everything else)
 * langford3.cu -- Godfrey's method with the FULL order-8 symmetry group.
 *
 * langford2 pinned x_{2n-1}=x_{2n}=+1 (the Z2xZ2 subgroup) and enumerated
 * 2^{2n-2} sign vectors.  The group is actually of order 8: reflection
 * rho: x_p -> x_{2n+1-p} also fixes (prod x)*F.  Reflection is not a
 * coordinate pinning, so it is exploited as follows.
 *
 *   sigma(b) = canon(reverse(b))     (canon = the unique element of
 *   {id, neg-all, neg-even, neg-odd} restoring b_{M-1}=b_{M-2}=0)
 *
 * sigma is an involution on the canonical set, preserves the summand, and
 * acts on (b_0,b_1) by swapping them (all three verified exhaustively by
 * refl_test.c for n=7,8,11).  Writing HI = bits [n,2n-3], LO = bits [2,n-1],
 * sigma maps LO onto HI bit-reversed (XORed with all-ones when b_0=b_1=1):
 *
 *   class (0,1) : sigma sends it to class (1,0)  -> enumerate one, weight 2
 *   class (0,0) : sigma stays inside             -> enumerate H <= W only
 *   class (1,1) : sigma stays inside             -> enumerate H <= W only
 *
 * with W = the image of LO.  Substituting W for LO as the loop coordinate
 * keeps the Gray code intact (the substitution is a bit permutation plus a
 * XOR, so one loop bit flipped is still one mask bit flipped).
 *
 * H <= W is enforced at SHARD level, not per element: the top t bits of H
 * and of W are fixed by the host, so the predicate is warp-uniform and the
 * lower triangle of shards is simply never launched.  Only the 2^t diagonal
 * shards (~0.2% of the work) need the per-element test.
 *
 * Net: 2^{2n-3} vectors instead of 2^{2n-2}.  Exactly 2x.
 * ===================================================================== */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#define CHECK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d: %s\n",#x,__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(1);} }while(0)

/* tables indexed by the MASK bit position that flips (0..2n-1) */
/* all tables are indexed by the LOOP bit and pre-permuted through psi at
 * build time, so the inner loop performs no dependent table indirection. */
__constant__ uint32_t cValidA[64];
__constant__ uint32_t cValidB[64];
__constant__ uint32_t cSub[64][8];
__constant__ int      cShiftJ[64];   /* psi[j]        */
__constant__ int      cShiftA[64];   /* psi[j] + 2    */
__constant__ int      cShiftB[64];   /* B-shift at psi[j] */
__constant__ uint64_t cBmask[64];    /* 1 << psi[j]   */
__constant__ uint64_t cRmask[64];    /* 1 << (M-1-psi[j]) */

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


/* ---- 160-bit product of the 32 SWAR lanes (sign included in the lanes) ---- */
__device__ __forceinline__ void prod160(const uint32_t *rr, uint32_t *z, int &neg)
{
    int p[16];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        uint32_t d = rr[k] ^ 0x80808080u;
        int v0 = (int)(signed char)(d);
        int v1 = (int)(signed char)(d >> 8);
        int v2 = (int)(signed char)(d >> 16);
        int v3 = (int)(signed char)(d >> 24);
        p[2*k]   = v0 * v1;
        p[2*k+1] = v2 * v3;
    }
    int q4[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) q4[k] = p[2*k] * p[2*k+1];
    long long s[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) s[k] = (long long)q4[2*k] * q4[2*k+1];
    neg = 0;
    unsigned long long u[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        neg ^= (s[k] < 0);
        u[k] = (unsigned long long)(s[k] < 0 ? -s[k] : s[k]);
    }
    uint32_t A[3], C[3];
    {   uint32_t a0=(uint32_t)u[0],a1=(uint32_t)(u[0]>>32);
        uint32_t c0=(uint32_t)u[1],c1=(uint32_t)(u[1]>>32);
        uint64_t p00=(uint64_t)a0*c0,p01=(uint64_t)a0*c1,
                 p10=(uint64_t)a1*c0,p11=(uint64_t)a1*c1;
        uint64_t m=(p00>>32)+(uint32_t)p01+(uint32_t)p10;
        A[0]=(uint32_t)p00; A[1]=(uint32_t)m;
        A[2]=(uint32_t)((m>>32)+(p01>>32)+(p10>>32)+p11); }
    {   uint32_t a0=(uint32_t)u[2],a1=(uint32_t)(u[2]>>32);
        uint32_t c0=(uint32_t)u[3],c1=(uint32_t)(u[3]>>32);
        uint64_t p00=(uint64_t)a0*c0,p01=(uint64_t)a0*c1,
                 p10=(uint64_t)a1*c0,p11=(uint64_t)a1*c1;
        uint64_t m=(p00>>32)+(uint32_t)p01+(uint32_t)p10;
        C[0]=(uint32_t)p00; C[1]=(uint32_t)m;
        C[2]=(uint32_t)((m>>32)+(p01>>32)+(p10>>32)+p11); }
    {   uint64_t P00=(uint64_t)A[0]*C[0];
        uint64_t P01=(uint64_t)A[0]*C[1], P10=(uint64_t)A[1]*C[0];
        uint64_t P02=(uint64_t)A[0]*C[2], P11=(uint64_t)A[1]*C[1], P20=(uint64_t)A[2]*C[0];
        uint64_t P12=(uint64_t)A[1]*C[2], P21=(uint64_t)A[2]*C[1];
        uint64_t P22=(uint64_t)A[2]*C[2];
        uint64_t cy, s0;
        s0=(uint32_t)P00;                          z[0]=(uint32_t)s0;
        cy=(s0>>32)+(P00>>32);
        s0=cy+(uint32_t)P01+(uint32_t)P10;         z[1]=(uint32_t)s0;
        cy=(s0>>32)+(P01>>32)+(P10>>32);
        s0=cy+(uint32_t)P02+(uint32_t)P11+(uint32_t)P20; z[2]=(uint32_t)s0;
        cy=(s0>>32)+(P02>>32)+(P11>>32)+(P20>>32);
        s0=cy+(uint32_t)P12+(uint32_t)P21;         z[3]=(uint32_t)s0;
        cy=(s0>>32)+(P12>>32)+(P21>>32);
        s0=cy+(uint32_t)P22;                       z[4]=(uint32_t)s0; }
}

template<int N, bool DIAG>
__global__ __launch_bounds__(256)
void godfrey3(uint64_t bbase, int innerBits, int kbits, uint32_t * __restrict__ out)
{
    const int M = 2*N;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* loop-space position must be 64-bit: 2k can exceed 32 bits */
    uint64_t g = (uint64_t)tid << innerBits;
    uint64_t b = bbase;
    for (uint64_t q = g; q; q &= q-1) b ^= cBmask[__ffsll((long long)q)-1];
    uint64_t rb = __brevll(b) >> (64 - M);

    uint32_t r[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) r[k] = 0x81818181u;
    #pragma unroll
    for (int i = 2; i <= N+1; i++) {
        const int w = M - i;
        const uint64_t mk = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
        int P = w - 2*__popcll((b ^ (b >> i)) & mk);
        const int l = i - 2;
        const int nl = (l & 1) ? (16 + ((l-1)>>1)) : (l>>1);   /* even lags first */
        r[nl>>2] = (r[nl>>2] & ~(0xFFu << (8*(nl&3)))) | ((uint32_t)(P+128) << (8*(nl&3)));
    }

    int sign = (__popcll(b) & 1) ? -1 : 1;
    uint32_t acc[5] = {0,0,0,0,0};
    const uint64_t kmask = (1ULL << kbits) - 1ULL;
    const uint64_t steps = 1ULL << innerBits;

    for (uint64_t t = 0; ; t++) {
        int w8 = 2;                              /* weight of this term */
        if (DIAG){
            uint32_t Hlo = (uint32_t)(g & kmask), Wlo = (uint32_t)(g >> kbits);
            w8 = (Hlo < Wlo) ? 2 : ((Hlo == Wlo) ? 1 : 0);
        }
        if (w8) {
            int p[16];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                uint32_t d = r[k] ^ 0x80808080u;
                int v0 = (int)(signed char)(d);
                int v1 = (int)(signed char)(d >> 8);
                int v2 = (int)(signed char)(d >> 16);
                int v3 = (int)(signed char)(d >> 24);
                p[2*k]   = v0 * v1;
                p[2*k+1] = v2 * v3;
            }
            int q4[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) q4[k] = p[2*k] * p[2*k+1];
            long long s[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) s[k] = (long long)q4[2*k] * q4[2*k+1];
            int neg = (sign < 0);
            unsigned long long u[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                neg ^= (s[k] < 0);
                u[k] = (unsigned long long)(s[k] < 0 ? -s[k] : s[k]);
            }
            uint32_t A[3], C[3];
            {   uint32_t a0=(uint32_t)u[0],a1=(uint32_t)(u[0]>>32);
                uint32_t c0=(uint32_t)u[1],c1=(uint32_t)(u[1]>>32);
                uint64_t p00=(uint64_t)a0*c0,p01=(uint64_t)a0*c1,
                         p10=(uint64_t)a1*c0,p11=(uint64_t)a1*c1;
                uint64_t m=(p00>>32)+(uint32_t)p01+(uint32_t)p10;
                A[0]=(uint32_t)p00; A[1]=(uint32_t)m;
                A[2]=(uint32_t)((m>>32)+(p01>>32)+(p10>>32)+p11); }
            {   uint32_t a0=(uint32_t)u[2],a1=(uint32_t)(u[2]>>32);
                uint32_t c0=(uint32_t)u[3],c1=(uint32_t)(u[3]>>32);
                uint64_t p00=(uint64_t)a0*c0,p01=(uint64_t)a0*c1,
                         p10=(uint64_t)a1*c0,p11=(uint64_t)a1*c1;
                uint64_t m=(p00>>32)+(uint32_t)p01+(uint32_t)p10;
                C[0]=(uint32_t)p00; C[1]=(uint32_t)m;
                C[2]=(uint32_t)((m>>32)+(p01>>32)+(p10>>32)+p11); }
            uint32_t z[5];
            {   uint64_t P00=(uint64_t)A[0]*C[0];
                uint64_t P01=(uint64_t)A[0]*C[1], P10=(uint64_t)A[1]*C[0];
                uint64_t P02=(uint64_t)A[0]*C[2], P11=(uint64_t)A[1]*C[1], P20=(uint64_t)A[2]*C[0];
                uint64_t P12=(uint64_t)A[1]*C[2], P21=(uint64_t)A[2]*C[1];
                uint64_t P22=(uint64_t)A[2]*C[2];
                uint64_t cy, s0;
                s0=(uint32_t)P00;                          z[0]=(uint32_t)s0;
                cy=(s0>>32)+(P00>>32);
                s0=cy+(uint32_t)P01+(uint32_t)P10;         z[1]=(uint32_t)s0;
                cy=(s0>>32)+(P01>>32)+(P10>>32);
                s0=cy+(uint32_t)P02+(uint32_t)P11+(uint32_t)P20; z[2]=(uint32_t)s0;
                cy=(s0>>32)+(P02>>32)+(P11>>32)+(P20>>32);
                s0=cy+(uint32_t)P12+(uint32_t)P21;         z[3]=(uint32_t)s0;
                cy=(s0>>32)+(P12>>32)+(P21>>32);
                s0=cy+(uint32_t)P22;                       z[4]=(uint32_t)s0; }
            /* weight 2 is applied once at the end of the thread (a single
             * 160-bit doubling) instead of a second 160-bit add per term. */
            if (neg) SUB160(acc,z); else ADD160(acc,z);
            if (DIAG && w8 == 2){ if (neg) SUB160(acc,z); else ADD160(acc,z); }
        }

        if (t + 1 == steps) break;

        const int j  = __ffsll((long long)(t + 1)) - 1;
        const uint32_t cm = -(uint32_t)((b >> cShiftJ[j]) & 1ULL);
        uint32_t Am = ((uint32_t)(b  >> cShiftA[j]) ^ cm) & cValidA[j];
        uint32_t Bm = ((uint32_t)(rb >> cShiftB[j]) ^ cm) & cValidB[j];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int sh = (k < 4) ? (8*k) : (8*(k-4) + 1);
            uint32_t ta = (((Am >> sh) & 0x55u) * 0x104104u) & 0x04040404u;
            uint32_t tb = (((Bm >> sh) & 0x55u) * 0x104104u) & 0x04040404u;
            r[k] = r[k] + ta + tb - cSub[j][k];
        }
        b  ^= cBmask[j];
        rb ^= cRmask[j];
        if (DIAG) g ^= 1ULL << j;
        sign = -sign;
    }

    if (!DIAG){                      /* every term had weight 2 */
        acc[4]=(acc[4]<<1)|(acc[3]>>31); acc[3]=(acc[3]<<1)|(acc[2]>>31);
        acc[2]=(acc[2]<<1)|(acc[1]>>31); acc[1]=(acc[1]<<1)|(acc[0]>>31);
        acc[0]<<=1;
    }
    __shared__ uint32_t sm[256][5];
    #pragma unroll
    for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = acc[k];
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint32_t x[5], y[5];
            #pragma unroll
            for (int k = 0; k < 5; k++){ x[k]=sm[threadIdx.x][k]; y[k]=sm[threadIdx.x+stride][k]; }
            ADD160(x,y);
            #pragma unroll
            for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = x[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        #pragma unroll
        for (int k = 0; k < 5; k++) out[blockIdx.x*5+k] = sm[0][k];
}


/* ===================== v4 fast path (non-diagonal shards, N<=31) ==========
 * Identical arithmetic to godfrey3<N,false>, but the 160-bit product is only
 * evaluated for terms that are not identically zero.  Survivors are compacted
 * warp-wide into a shared ring buffer and drained 32 at a time, so every
 * product runs on a full warp: no divergence penalty.
 * ======================================================================== */
template<int N>
__global__ __launch_bounds__(256)
void godfrey4(uint64_t bbase, int innerBits, int kbits, uint32_t * __restrict__ out)
{
    const int M = 2*N;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t g = (uint64_t)tid << innerBits;
    uint64_t b = bbase;
    for (uint64_t q = g; q; q &= q-1) b ^= cBmask[__ffsll((long long)q)-1];
    uint64_t rb = __brevll(b) >> (64 - M);

    uint32_t r[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) r[k] = 0x81818181u;
    #pragma unroll
    for (int i = 2; i <= N+1; i++) {
        const int w = M - i;
        const uint64_t mk = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
        int P = w - 2*__popcll((b ^ (b >> i)) & mk);
        const int l = i - 2;
        const int nl = (l & 1) ? (16 + ((l-1)>>1)) : (l>>1);
        r[nl>>2] = (r[nl>>2] & ~(0xFFu << (8*(nl&3)))) | ((uint32_t)(P+128) << (8*(nl&3)));
    }
    /* lane 31 is unused when N<=31: the term's sign rides there as +-1 */
    if (__popcll(b) & 1) r[7] ^= 0xFE000000u;

    uint32_t acc[5] = {0,0,0,0,0};
    const uint64_t steps = 1ULL << innerBits;

    __shared__ uint4 sq[8][64][2];             /* [warp][slot] ring, 32B/slot */
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const unsigned ltm = (1u << lane) - 1u;
    int head = 0, qlen = 0;                    /* warp-uniform */

    for (uint64_t t = 0; ; t++) {
        /* zero factor <=> some even-lag byte equals 0x80; even lags = words 0..3 */
        uint32_t hz;
        { uint32_t d = r[0] ^ 0x80808080u; hz  = (d - 0x01010101u) & ~d & 0x80808080u; }
        #pragma unroll
        for (int k = 1; k < 4; k++) {
            uint32_t d = r[k] ^ 0x80808080u; hz |= (d - 0x01010101u) & ~d & 0x80808080u;
        }
        const bool live = (hz == 0u);
        const unsigned bal = __ballot_sync(0xffffffffu, live);
        if (live) {
            const int s = (head + qlen + __popc(bal & ltm)) & 63;
            sq[warp][s][0] = make_uint4(r[0],r[1],r[2],r[3]);
            sq[warp][s][1] = make_uint4(r[4],r[5],r[6],r[7]);
        }
        qlen += __popc(bal);
        if (qlen >= 32) {
            __syncwarp();
            const int s = (head + lane) & 63;
            const uint4 e0 = sq[warp][s][0], e1 = sq[warp][s][1];
            const uint32_t qr[8] = {e0.x,e0.y,e0.z,e0.w,e1.x,e1.y,e1.z,e1.w};
            uint32_t z[5]; int neg; prod160(qr, z, neg);
            if (neg) SUB160(acc,z); else ADD160(acc,z);
            head = (head + 32) & 63; qlen -= 32;
        }

        if (t + 1 == steps) break;

        const int j  = __ffsll((long long)(t + 1)) - 1;
        const uint32_t cm = -(uint32_t)((b >> cShiftJ[j]) & 1ULL);
        uint32_t Am = ((uint32_t)(b  >> cShiftA[j]) ^ cm) & cValidA[j];
        uint32_t Bm = ((uint32_t)(rb >> cShiftB[j]) ^ cm) & cValidB[j];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int sh = (k < 4) ? (8*k) : (8*(k-4) + 1);
            uint32_t ta = (((Am >> sh) & 0x55u) * 0x104104u) & 0x04040404u;
            uint32_t tb = (((Bm >> sh) & 0x55u) * 0x104104u) & 0x04040404u;
            r[k] = r[k] + ta + tb - cSub[j][k];
        }
        b  ^= cBmask[j];
        rb ^= cRmask[j];
        r[7] ^= 0xFE000000u;                   /* sign toggle, lane 31 */
    }

    __syncwarp();
    if (lane < qlen) {                          /* drain the tail */
        const int s = (head + lane) & 63;
        const uint4 e0 = sq[warp][s][0], e1 = sq[warp][s][1];
        const uint32_t qr[8] = {e0.x,e0.y,e0.z,e0.w,e1.x,e1.y,e1.z,e1.w};
        uint32_t z[5]; int neg; prod160(qr, z, neg);
        if (neg) SUB160(acc,z); else ADD160(acc,z);
    }

    /* every term of a non-diagonal shard carries weight 2 */
    acc[4]=(acc[4]<<1)|(acc[3]>>31); acc[3]=(acc[3]<<1)|(acc[2]>>31);
    acc[2]=(acc[2]<<1)|(acc[1]>>31); acc[1]=(acc[1]<<1)|(acc[0]>>31);
    acc[0]<<=1;

    __syncthreads();                            /* reuse sq for the reduction */
    uint32_t (*sm)[5] = (uint32_t(*)[5])&sq[0][0][0].x;
    #pragma unroll
    for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = acc[k];
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint32_t x[5], y[5];
            #pragma unroll
            for (int k = 0; k < 5; k++){ x[k]=sm[threadIdx.x][k]; y[k]=sm[threadIdx.x+stride][k]; }
            ADD160(x,y);
            #pragma unroll
            for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = x[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        #pragma unroll
        for (int k = 0; k < 5; k++) out[blockIdx.x*5+k] = sm[0][k];
}


/* ===================== v5 fast path (non-diagonal shards, N<=31) ==========
 * v4 kept all 31 autocorrelations live in 8 SWAR registers, but only the 16
 * EVEN lags can ever vanish, and 87.1% of the vectors die on them.  v5 keeps
 * only those 16 (4 registers, half the update) and rebuilds the 15 odd ones
 * from b at drain time -- paid on 12.9% of the vectors instead of 100%.
 * The Gray index j is 0,1,0,dynamic over a 4-step block, so 3 steps out of 4
 * address the constant bank with a literal offset (no indexed load, no FLO).
 * ======================================================================== */
template<int N>
__device__ __forceinline__ void expand_odd(uint64_t b, uint32_t *q)
{
    const int M = 2*N;
    q[4]=q[5]=q[6]=q[7]=0x81818181u;
    #pragma unroll
    for (int i = 3; i <= N+1; i += 2) {
        const int w = M - i;
        const uint64_t mk = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
        int P = w - 2*__popcll((b ^ (b >> i)) & mk);
        const int nl = 16 + ((i-3)>>1);
        q[nl>>2] = (q[nl>>2] & ~(0xFFu << (8*(nl&3)))) | ((uint32_t)(P+128) << (8*(nl&3)));
    }
    if (__popcll(b) & 1) q[7] ^= 0xFE000000u;      /* sign rides in lane 31 */
}

#define LF5_UPD(J) do {                                                        \
    const uint32_t cm = -(uint32_t)((b >> cShiftJ[J]) & 1ULL);                 \
    const uint32_t Am = ((uint32_t)(b  >> cShiftA[J]) ^ cm) & cValidA[J];      \
    const uint32_t Bm = ((uint32_t)(rb >> cShiftB[J]) ^ cm) & cValidB[J];      \
    r[0] = r[0] + ((((Am     )&0x55u)*0x104104u)&0x04040404u)                  \
                + ((((Bm     )&0x55u)*0x104104u)&0x04040404u) - cSub[J][0];    \
    r[1] = r[1] + ((((Am >> 8)&0x55u)*0x104104u)&0x04040404u)                  \
                + ((((Bm >> 8)&0x55u)*0x104104u)&0x04040404u) - cSub[J][1];    \
    r[2] = r[2] + ((((Am >>16)&0x55u)*0x104104u)&0x04040404u)                  \
                + ((((Bm >>16)&0x55u)*0x104104u)&0x04040404u) - cSub[J][2];    \
    r[3] = r[3] + ((((Am >>24)&0x55u)*0x104104u)&0x04040404u)                  \
                + ((((Bm >>24)&0x55u)*0x104104u)&0x04040404u) - cSub[J][3];    \
    b ^= cBmask[J]; rb ^= cRmask[J];                                           \
  } while(0)

#define LF5_POINT() do {                                                       \
    uint32_t d0=r[0]^0x80808080u, d1=r[1]^0x80808080u,                         \
             d2=r[2]^0x80808080u, d3=r[3]^0x80808080u;                         \
    uint32_t hz = ((d0-0x01010101u)&~d0&0x80808080u)                           \
                | ((d1-0x01010101u)&~d1&0x80808080u)                           \
                | ((d2-0x01010101u)&~d2&0x80808080u)                           \
                | ((d3-0x01010101u)&~d3&0x80808080u);                          \
    const unsigned bal = __ballot_sync(0xffffffffu, hz==0u);                   \
    if (hz==0u) { const int s=(head+qlen+__popc(bal&ltm))&63;                  \
        sq[warp][s][0]=make_uint4(r[0],r[1],r[2],r[3]);                        \
        sq[warp][s][1]=make_uint4((uint32_t)b,(uint32_t)(b>>32),0,0); }        \
    qlen += __popc(bal);                                                       \
    if (qlen >= 32) { __syncwarp();                                            \
        const int s=(head+lane)&63;                                            \
        const uint4 e0=sq[warp][s][0], e1=sq[warp][s][1];                      \
        uint32_t q[8]; q[0]=e0.x;q[1]=e0.y;q[2]=e0.z;q[3]=e0.w;                \
        expand_odd<N>(((uint64_t)e1.y<<32)|(uint64_t)e1.x, q);                 \
        uint32_t z5[5]; int neg; prod160(q,z5,neg);                            \
        if (neg) SUB160(acc,z5); else ADD160(acc,z5);                          \
        head=(head+32)&63; qlen-=32; }                                         \
  } while(0)

template<int N>
__global__ __launch_bounds__(256,6)
void godfrey5(uint64_t bbase, int innerBits, int kbits, uint32_t * __restrict__ out)
{
    const int M = 2*N;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t g = (uint64_t)tid << innerBits;
    uint64_t b = bbase;
    for (uint64_t z = g; z; z &= z-1) b ^= cBmask[__ffsll((long long)z)-1];
    uint64_t rb = __brevll(b) >> (64 - M);

    uint32_t r[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) r[k] = 0x81818181u;
    #pragma unroll
    for (int i = 2; i <= N+1; i += 2) {                 /* even lags only */
        const int w = M - i;
        const uint64_t mk = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
        int P = w - 2*__popcll((b ^ (b >> i)) & mk);
        const int nl = (i-2)>>1;
        r[nl>>2] = (r[nl>>2] & ~(0xFFu << (8*(nl&3)))) | ((uint32_t)(P+128) << (8*(nl&3)));
    }

    uint32_t acc[5] = {0,0,0,0,0};
    __shared__ uint4 sq[8][64][2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned ltm = (1u << lane) - 1u;
    int head = 0, qlen = 0;

    const uint64_t S = (1ULL << innerBits) >> 3;        /* blocks of 8 steps */
    for (uint64_t s = 0; ; s++) {
        LF5_POINT(); LF5_UPD(0);
        LF5_POINT(); LF5_UPD(1);
        LF5_POINT(); LF5_UPD(0);
        LF5_POINT(); LF5_UPD(2);
        LF5_POINT(); LF5_UPD(0);
        LF5_POINT(); LF5_UPD(1);
        LF5_POINT(); LF5_UPD(0);
        LF5_POINT();
        if (s + 1 == S) break;
        const int j = 3 + (__ffsll((long long)(s + 1)) - 1);
        LF5_UPD(j);
    }

    __syncwarp();
    if (lane < qlen) {
        const int s = (head + lane) & 63;
        const uint4 e0 = sq[warp][s][0], e1 = sq[warp][s][1];
        uint32_t q[8]; q[0]=e0.x;q[1]=e0.y;q[2]=e0.z;q[3]=e0.w;
        expand_odd<N>(((uint64_t)e1.y<<32)|(uint64_t)e1.x, q);
        uint32_t z5[5]; int neg; prod160(q,z5,neg);
        if (neg) SUB160(acc,z5); else ADD160(acc,z5);
    }

    acc[4]=(acc[4]<<1)|(acc[3]>>31); acc[3]=(acc[3]<<1)|(acc[2]>>31);
    acc[2]=(acc[2]<<1)|(acc[1]>>31); acc[1]=(acc[1]<<1)|(acc[0]>>31);
    acc[0]<<=1;

    __syncthreads();
    uint32_t (*sm)[5] = (uint32_t(*)[5])&sq[0][0][0].x;
    #pragma unroll
    for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = acc[k];
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint32_t x[5], y[5];
            #pragma unroll
            for (int k = 0; k < 5; k++){ x[k]=sm[threadIdx.x][k]; y[k]=sm[threadIdx.x+stride][k]; }
            ADD160(x,y);
            #pragma unroll
            for (int k = 0; k < 5; k++) sm[threadIdx.x][k] = x[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        #pragma unroll
        for (int k = 0; k < 5; k++) out[blockIdx.x*5+k] = sm[0][k];
}

/* ====================== host ====================== */
typedef struct { uint32_t w[5]; } u160;
static void h_add(u160 *z,const u160 *x){ uint64_t c=0;
    for(int k=0;k<5;k++){ uint64_t s=(uint64_t)z->w[k]+x->w[k]+c; z->w[k]=(uint32_t)s; c=s>>32; } }
static void h_shl2(u160 *z){ uint32_t c=0;
    for(int k=0;k<5;k++){ uint32_t v=z->w[k]; z->w[k]=(v<<2)|c; c=v>>30; } }
static void print_u160(u160 v){ char o[64]; int len=0; uint32_t t[5]; memcpy(t,v.w,sizeof t);
    for(;;){ uint64_t rem=0;
        for(int k=4;k>=0;k--){ uint64_t c=(rem<<32)|t[k]; t[k]=(uint32_t)(c/1000000000ULL); rem=c%1000000000ULL; }
        int nz=0; for(int k=0;k<5;k++) if(t[k]) nz=1;
        for(int d=0;d<9;d++){ o[len++]='0'+(int)(rem%10); rem/=10; }
        if(!nz) break; }
    while(len>1&&o[len-1]=='0') len--;
    for(int k=len-1;k>=0;k--) putchar(o[k]); }
static void print_i160(u160 v){ if(v.w[4]&0x80000000u){ putchar('-'); uint64_t c=1;
        for(int k=0;k<5;k++){ uint64_t s=(uint64_t)(~v.w[k])+c; v.w[k]=(uint32_t)s; c=s>>32; } }
    print_u160(v); }

typedef void (*kern_t)(uint64_t,int,int,uint32_t*);
/* godfrey4 needs a free SWAR lane for the sign, which exists only for N<=31 */
template<int NN> struct fast_t  { static kern_t get(){ return (kern_t)godfrey5<NN>; } };
template<> struct fast_t<32>    { static kern_t get(){ return (kern_t)godfrey3<32,false>; } };
template<int NN> struct fast4_t { static kern_t get(){ return (kern_t)godfrey4<NN>; } };
template<> struct fast4_t<32>   { static kern_t get(){ return (kern_t)godfrey3<32,false>; } };
/* godfrey5 unrolls 8 Gray steps, so it needs innerBits >= 3; the smallest n
 * cannot afford that (threadBits would drop below 8) and fall back to v4. */
#define INST(NN) case NN: return diag ? (kern_t)godfrey3<NN,true> \
                                      : (v5 ? fast_t<NN>::get() : fast4_t<NN>::get());
static kern_t pick(int N,bool diag,bool v5){ switch(N){
    INST(7) INST(8) INST(11) INST(12) INST(15) INST(16) INST(19) INST(20)
    INST(23) INST(24) INST(27) INST(28) INST(31) INST(32) default: return NULL; } }

static void build_tables(int N,int k){
    const int M=2*N;
    int psi[64]; for(int i=0;i<64;i++) psi[i]=0;
    for(int i=0;i<k;i++) psi[i]     = N + i;        /* H low bits */
    for(int j=0;j<k;j++) psi[k+j]   = N - 1 - j;    /* W low bits */

    uint32_t hA[64]={0},hB[64]={0},hSub[64][8]; memset(hSub,0,sizeof hSub);
    int hSJ[64]={0},hSA[64]={0},hSB[64]={0};
    uint64_t hBm[64]={0},hRm[64]={0};
    for(int j=0;j<2*k;j++){
        int p=psi[j];                               /* mask bit that flips */
        int ca=M-2-p; if(ca>N) ca=N; if(ca<0) ca=0;
        int cb=p-1;   if(cb>N) cb=N; if(cb<0) cb=0;
        hA[j]=(ca>=32)?0xFFFFFFFFu:((1u<<ca)-1);
        hB[j]=(cb>=32)?0xFFFFFFFFu:((1u<<cb)-1);
        hSJ[j]=p; hSA[j]=p+2; hSB[j]=(p>=2)?(M+1-p):0;
        hBm[j]=1ULL<<p; hRm[j]=1ULL<<(M-1-p);
        for(int l=0;l<32;l++){ uint32_t v=2*((hA[j]>>l)&1)+2*((hB[j]>>l)&1);
                               int nl=(l&1)?(16+((l-1)>>1)):(l>>1);
                               hSub[j][nl>>2]|=v<<(8*(nl&3)); } }
    CHECK(cudaMemcpyToSymbol(cValidA,hA,sizeof hA));
    CHECK(cudaMemcpyToSymbol(cValidB,hB,sizeof hB));
    CHECK(cudaMemcpyToSymbol(cSub,hSub,sizeof hSub));
    CHECK(cudaMemcpyToSymbol(cShiftJ,hSJ,sizeof hSJ));
    CHECK(cudaMemcpyToSymbol(cShiftA,hSA,sizeof hSA));
    CHECK(cudaMemcpyToSymbol(cShiftB,hSB,sizeof hSB));
    CHECK(cudaMemcpyToSymbol(cBmask,hBm,sizeof hBm));
    CHECK(cudaMemcpyToSymbol(cRmask,hRm,sizeof hRm));
}

int main(int argc,char**argv){
    int N=12, tbits=-1, innerBits=-1, chunkTarget=34;
    long long from=0,count=-1; const char*ckpt=NULL;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-n")) N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--t")) tbits=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--inner")) innerBits=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--chunklog")) chunkTarget=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--from")) from=atoll(argv[++i]);
        else if(!strcmp(argv[i],"--count")) count=atoll(argv[++i]);
        else if(!strcmp(argv[i],"--ckpt")) ckpt=argv[++i];
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; } }

    const int M=2*N, nc=N-2;
    if(tbits<0){ int k=nc; while(2*k>chunkTarget) k--; tbits=nc-k; }
    const int k=nc-tbits;
    if(k<5||tbits<0){ fprintf(stderr,"bad split k=%d t=%d\n",k,tbits); return 1; }
    if(innerBits<0) innerBits = (2*k-8>14)?14:(2*k-8);
    if(innerBits<1) innerBits=1;
    /* godfrey5 wants innerBits>=3; take it when the layout allows it */
    if(innerBits<3 && 2*k-3>=8) innerBits=3;
    const int threadBits=2*k-innerBits;
    if(threadBits<8){ fprintf(stderr,"n too small for this layout\n"); return 1; }
    const int blocks=1<<(threadBits-8);
    const long long T=1LL<<tbits;
    const long long nShards = T*T + 2*(T*(T+1)/2);

    build_tables(N,k);
    uint32_t*d_out; CHECK(cudaMalloc(&d_out,(size_t)blocks*5*4));
    uint32_t*h_out=(uint32_t*)malloc((size_t)blocks*5*4);
    if(count<0) count=nShards-from;

    /* class base masks */
    uint64_t loMask=0; for(int p=2;p<=N-1;p++) loMask|=1ULL<<p;
    uint64_t baseCls[3]; baseCls[0]=(1ULL<<1);            /* A = (b0,b1)=(0,1) */
    baseCls[1]=0;                                          /* B = (0,0)        */
    baseCls[2]=(1ULL<<0)|(1ULL<<1)|loMask;                 /* C = (1,1), LO complemented */

    fprintf(stderr,"n=%d  M=%d  coord bits=%d  split t=%d k=%d  shard=2^%d sums  "
                   "blocks=%d inner=2^%d\nshards=%lld  vectors=2^%d (half of 2^%d)\n",
            N,M,nc,tbits,k,2*k,blocks,innerBits,nShards,2*N-3,2*N-2);
    fprintf(stderr,"kernel: %s\n", innerBits>=3 ? "godfrey5 (half-state, unroll 8)" : "godfrey4 (fallback)");

    u160 total; memset(&total,0,sizeof total); long long done=0;
    if(ckpt){ FILE*f=fopen(ckpt,"r");
        if(f){ int n2,t2; long long fr,ct,dn; u160 s;
            if(fscanf(f,"ckpt3 %d %d %lld %lld %lld %x %x %x %x %x",&n2,&t2,&fr,&ct,&dn,
                      &s.w[0],&s.w[1],&s.w[2],&s.w[3],&s.w[4])==10
               && n2==N&&t2==tbits&&fr==from&&ct==count){ total=s; done=dn;
                 fprintf(stderr,"resumed: %lld/%lld shards done\n",done,count); }
            fclose(f);} }

    struct timespec T0,T1; clock_gettime(CLOCK_MONOTONIC,&T0); double last=0;
    long long idx=0, ran=0;
    for(int cls=0; cls<3; cls++){
        for(long long a=0; a<T; a++){
            for(long long bb=(cls==0?0:a); bb<T; bb++){
                if(idx < from + done){ idx++; continue; }
                if(idx >= from + count){ cls=3; a=T; break; }
                bool diag = (cls!=0) && (a==bb);
                uint64_t base = baseCls[cls];
                for(int i=0;i<tbits;i++) if((a>>i)&1) base ^= 1ULL<<(N+k+i);
                for(int j=0;j<tbits;j++) if((bb>>j)&1) base ^= 1ULL<<(N-1-k-j);
                kern_t K=pick(N,diag,innerBits>=3);
                K<<<blocks,256>>>(base,innerBits,k,d_out);
                CHECK(cudaGetLastError());
                CHECK(cudaMemcpy(h_out,d_out,(size_t)blocks*5*4,cudaMemcpyDeviceToHost));
                for(int q=0;q<blocks;q++) h_add(&total,(u160*)(h_out+5*q));
                idx++; done++; ran++;
                clock_gettime(CLOCK_MONOTONIC,&T1);
                double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)*1e-9;
                if(el-last>10.0||done==count){ last=el;
                    fprintf(stderr,"  %lld/%lld shards  %.3f Gsums/s  elapsed %.2f h  ETA %.2f h\n",
                            done,count,(double)ran*(double)(1ULL<<(2*k))/el*1e-9,
                            el/3600.0,(count-done)*el/ran/3600.0);
                    if(ckpt){ char tmp[512]; snprintf(tmp,sizeof tmp,"%s.tmp",ckpt);
                        FILE*f=fopen(tmp,"w");
                        if(f){ fprintf(f,"ckpt3 %d %d %lld %lld %lld %08x %08x %08x %08x %08x\n",
                                       N,tbits,from,count,done,total.w[0],total.w[1],
                                       total.w[2],total.w[3],total.w[4]);
                               fclose(f); rename(tmp,ckpt);} } } } } }

    clock_gettime(CLOCK_MONOTONIC,&T1);
    double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)*1e-9;
    fprintf(stderr,"ran %lld shards, %.6g vectors in %.1f s -> %.4f Gsums/s\n",
            ran,(double)ran*(double)(1ULL<<(2*k)),el,(double)ran*(double)(1ULL<<(2*k))/el*1e-9);

    if(from==0&&count==nShards){
        h_shl2(&total);
        int ok=1; for(int bit=0;bit<M;bit++) if((total.w[bit>>5]>>(bit&31))&1) ok=0;
        if(!ok){ fprintf(stderr,"*** SELF-CHECK FAILED: not divisible by 2^%d ***\n",M); return 1; }
        u160 v; memset(&v,0,sizeof v);
        for(int bit=M;bit<160;bit++) if((total.w[bit>>5]>>(bit&31))&1) v.w[(bit-M)>>5]|=1u<<((bit-M)&31);
        printf("n=%d   V(n) = 2*L(2,n) = ",N); print_u160(v); printf("\n");
        uint32_t c=0; u160 h;
        for(int q=4;q>=0;q--){ uint32_t x=v.w[q]; h.w[q]=(x>>1)|(c<<31); c=x&1; }
        printf("n=%d   L(2,%d)         = ",N,N); print_u160(h); printf("\n");
    } else { printf("PARTIAL n=%d t=%d shards %lld..%lld sum = ",N,tbits,from,from+count-1);
             print_i160(total); printf("\n"); }
    return 0;
}
