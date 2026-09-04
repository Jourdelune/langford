#include <time.h>
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

/* ====================== host side ====================== */
typedef struct { uint32_t w[5]; } u160;
static void h_add(u160 *z, const u160 *x){
    uint64_t c = 0;
    for (int k = 0; k < 5; k++){ uint64_t s = (uint64_t)z->w[k] + x->w[k] + c; z->w[k]=(uint32_t)s; c = s>>32; }
}
static void h_shl2(u160 *z){
    uint32_t carry = 0;
    for (int k = 0; k < 5; k++){ uint32_t v = z->w[k]; z->w[k] = (v<<2)|carry; carry = v>>30; }
}
static void print_u160(u160 v){
    char out[64]; int len = 0; uint32_t t[5]; memcpy(t, v.w, sizeof t);
    for (;;){
        uint64_t rem = 0;
        for (int k = 4; k >= 0; k--){ uint64_t cur = (rem<<32) | t[k]; t[k]=(uint32_t)(cur/1000000000ULL); rem = cur%1000000000ULL; }
        int nz = 0; for (int k = 0; k < 5; k++) if (t[k]) nz = 1;
        for (int d = 0; d < 9; d++){ out[len++] = '0' + (int)(rem%10); rem/=10; }
        if (!nz) break;
    }
    while (len > 1 && out[len-1]=='0') len--;
    for (int k = len-1; k >= 0; k--) putchar(out[k]);
}
static void print_i160(u160 v){                 /* two's complement, signed */
    if (v.w[4] & 0x80000000u){
        putchar('-');
        uint64_t c = 1;
        for (int k = 0; k < 5; k++){ uint64_t s = (uint64_t)(~v.w[k]) + c; v.w[k]=(uint32_t)s; c = s>>32; }
    }
    print_u160(v);
}

typedef void (*kern_t)(uint64_t, int, uint32_t*);
static kern_t pick(int N){
    switch(N){
      case 7:  return godfrey<7>;   case 8:  return godfrey<8>;
      case 11: return godfrey<11>;  case 12: return godfrey<12>;
      case 15: return godfrey<15>;  case 16: return godfrey<16>;
      case 19: return godfrey<19>;  case 20: return godfrey<20>;
      case 23: return godfrey<23>;  case 24: return godfrey<24>;
      case 27: return godfrey<27>;  case 28: return godfrey<28>;
      case 31: return godfrey<31>;  case 32: return godfrey<32>;
      default: return NULL;
    }
}
static void build_tables(int N, int innerBits){
    const int M = 2*N;
    uint32_t hA[32]={0}, hB[32]={0}, hSub[32][8]; int hSh[32]={0};
    memset(hSub, 0, sizeof hSub);
    for (int j = 0; j < innerBits; j++){
        int ca = M - 2 - j;  if (ca > N) ca = N;  if (ca < 0) ca = 0;
        int cb = j - 1;      if (cb > N) cb = N;  if (cb < 0) cb = 0;
        hA[j] = (ca >= 32) ? 0xFFFFFFFFu : ((1u << ca) - 1);
        hB[j] = (cb >= 32) ? 0xFFFFFFFFu : ((1u << cb) - 1);
        hSh[j] = (j >= 2) ? (M + 1 - j) : 0;
        for (int l = 0; l < 32; l++){
            uint32_t v = 2*((hA[j]>>l)&1) + 2*((hB[j]>>l)&1);
            hSub[j][l>>2] |= v << (8*(l&3));
        }
    }
    CHECK(cudaMemcpyToSymbol(cValidA, hA, sizeof hA));
    CHECK(cudaMemcpyToSymbol(cValidB, hB, sizeof hB));
    CHECK(cudaMemcpyToSymbol(cSub,  hSub, sizeof hSub));
    CHECK(cudaMemcpyToSymbol(cShiftB, hSh, sizeof hSh));
}

/* ---- final arithmetic: raw sum over 2^{2n-2} vectors  ->  L(2,n) ---- */
static int finish(int N, u160 total){
    const int M = 2*N;
    h_shl2(&total);                                  /* Z2 x Z2 symmetry: x4 */
    int ok = 1;
    for (int bit = 0; bit < M; bit++) if ((total.w[bit>>5] >> (bit&31)) & 1) ok = 0;
    if (!ok){ fprintf(stderr,"*** SELF-CHECK FAILED: sum not divisible by 2^%d ***\n", M); return 1; }
    u160 v; memset(&v,0,sizeof v);
    for (int bit = M; bit < 160; bit++)
        if ((total.w[bit>>5]>>(bit&31)) & 1) v.w[(bit-M)>>5] |= 1u << ((bit-M)&31);
    printf("n=%d   V(n) = 2*L(2,n) = ", N); print_u160(v); printf("\n");
    uint32_t carry = 0; u160 half;
    for (int k = 4; k >= 0; k--){ uint32_t x = v.w[k]; half.w[k]=(x>>1)|(carry<<31); carry = x&1; }
    printf("n=%d   L(2,%d)         = ", N, N); print_u160(half); printf("\n");
    return 0;
}

/* ---------------- partial-result files (distributed runs) ------------- */
static void write_part(const char *path, int N, int chunkLog, long long from,
                       long long count, u160 s, double secs){
    FILE *f = fopen(path,"w");
    if (!f){ perror("write_part"); return; }
    fprintf(f,"langford-partial 1\nn %d\nchunklog %d\nfrom %lld\ncount %lld\n"
              "sum %08x %08x %08x %08x %08x\nseconds %.3f\n",
            N, chunkLog, from, count, s.w[0],s.w[1],s.w[2],s.w[3],s.w[4], secs);
    fclose(f);
}

static int do_merge(int argc, char **argv, int first){
    int N = -1, chunkLog = -1;
    long long totalChunks = 0;
    static unsigned char *seen = NULL;
    u160 total; memset(&total,0,sizeof total);
    for (int a = first; a < argc; a++){
        FILE *f = fopen(argv[a],"r");
        if (!f){ fprintf(stderr,"cannot open %s\n", argv[a]); return 1; }
        int n2, cl; long long fr, ct; u160 s; double secs; char magic[64]; int ver;
        if (fscanf(f,"%63s %d",magic,&ver)!=2 || strcmp(magic,"langford-partial")){
            fprintf(stderr,"%s: not a partial file\n",argv[a]); return 1; }
        if (fscanf(f," n %d chunklog %d from %lld count %lld sum %x %x %x %x %x seconds %lf",
                   &n2,&cl,&fr,&ct,&s.w[0],&s.w[1],&s.w[2],&s.w[3],&s.w[4],&secs)!=10){
            fprintf(stderr,"%s: parse error\n",argv[a]); return 1; }
        fclose(f);
        if (N < 0){ N = n2; chunkLog = cl; totalChunks = 1LL << (2*N-2-chunkLog);
                    seen = (unsigned char*)calloc(totalChunks,1); }
        if (n2 != N || cl != chunkLog){ fprintf(stderr,"%s: mismatched n/chunklog\n",argv[a]); return 1; }
        for (long long c = fr; c < fr+ct; c++){
            if (c < 0 || c >= totalChunks){ fprintf(stderr,"%s: chunk %lld out of range\n",argv[a],c); return 1; }
            if (seen[c]){ fprintf(stderr,"%s: chunk %lld counted twice\n",argv[a],c); return 1; }
            seen[c] = 1;
        }
        h_add(&total,&s);
        fprintf(stderr,"  + %s : chunks %lld..%lld\n", argv[a], fr, fr+ct-1);
    }
    long long missing = 0;
    for (long long c = 0; c < totalChunks; c++) if (!seen[c]) missing++;
    if (missing){ fprintf(stderr,"INCOMPLETE: %lld of %lld chunks missing\n", missing, totalChunks);
                  printf("partial sum = "); print_i160(total); printf("\n"); return 1; }
    fprintf(stderr,"all %lld chunks present\n", totalChunks);
    return finish(N, total);
}

int main(int argc, char **argv){
    int N = 12, innerBits = 14, chunkLog = -1;
    long long from = 0, count = -1;
    const char *ckpt = NULL, *part = NULL;
    for (int i = 1; i < argc; i++){
        if (!strcmp(argv[i],"--merge"))         return do_merge(argc, argv, i+1);
        else if (!strcmp(argv[i],"-n"))         N = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--inner"))    innerBits = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--chunklog")) chunkLog = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--from"))     from = atoll(argv[++i]);
        else if (!strcmp(argv[i],"--count"))    count = atoll(argv[++i]);
        else if (!strcmp(argv[i],"--ckpt"))     ckpt = argv[++i];
        else if (!strcmp(argv[i],"--part"))     part = argv[++i];
        else { fprintf(stderr,"unknown arg %s\n", argv[i]); return 1; }
    }
    kern_t K = pick(N);
    if (!K){ fprintf(stderr,"n=%d not instantiated\n", N); return 1; }

    const int Lfree = 2*N - 2;
    if (chunkLog < 0) chunkLog = (Lfree < 34) ? Lfree : 34;   /* ~0.5 s per launch */
    if (chunkLog > Lfree) chunkLog = Lfree;
    while (innerBits + 8 > chunkLog) innerBits--;             /* need >=1 block */
    const int blocksLog = chunkLog - innerBits - 8;
    const int blocks = 1 << blocksLog;
    const long long totalChunks = 1LL << (Lfree - chunkLog);
    if (count < 0) count = totalChunks - from;

    build_tables(N, innerBits);
    uint32_t *d_out; CHECK(cudaMalloc(&d_out,(size_t)blocks*5*sizeof(uint32_t)));
    uint32_t *h_out = (uint32_t*)malloc((size_t)blocks*5*sizeof(uint32_t));

    u160 total; memset(&total,0,sizeof total);
    long long done = 0;

    /* resume */
    if (ckpt){
        FILE *f = fopen(ckpt,"r");
        if (f){
            int n2,cl; long long fr,ct,dn; u160 s;
            if (fscanf(f,"ckpt %d %d %lld %lld %lld %x %x %x %x %x",&n2,&cl,&fr,&ct,&dn,
                       &s.w[0],&s.w[1],&s.w[2],&s.w[3],&s.w[4])==10
                && n2==N && cl==chunkLog && fr==from && ct==count){
                total = s; done = dn;
                fprintf(stderr,"resuming from checkpoint: %lld/%lld chunks already done\n", done, count);
            } else fprintf(stderr,"checkpoint present but does not match this job; ignoring\n");
            fclose(f);
        }
    }

    fprintf(stderr,"n=%d  2n=%d  free bits=%d  chunk=2^%d sums  blocks=%d inner=2^%d\n"
                   "total chunks=%lld  running %lld..%lld\n",
            N, 2*N, Lfree, chunkLog, blocks, innerBits, totalChunks, from+done, from+count-1);

    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double secs0 = 0;
    for (long long c = from + done; c < from + count; c++){
        uint64_t base = (uint64_t)c << chunkLog;
        K<<<blocks,256>>>(base, innerBits, d_out);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_out, d_out, (size_t)blocks*5*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int k = 0; k < blocks; k++) h_add(&total,(u160*)(h_out+5*k));
        done++;
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double el = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
        if (el - secs0 > 10.0 || done == count){
            secs0 = el;
            double rate = (double)done*(double)(1ULL<<chunkLog)/el;
            double eta  = (count-done)*el/done;
            fprintf(stderr,"  %lld/%lld chunks  %.3f Gsums/s  elapsed %.1f h  ETA %.1f h\n",
                    done, count, rate*1e-9, el/3600.0, eta/3600.0);
            if (ckpt){
                char tmp[512]; snprintf(tmp,sizeof tmp,"%s.tmp",ckpt);
                FILE *f = fopen(tmp,"w");
                if (f){ fprintf(f,"ckpt %d %d %lld %lld %lld %08x %08x %08x %08x %08x\n",
                                N,chunkLog,from,count,done,
                                total.w[0],total.w[1],total.w[2],total.w[3],total.w[4]);
                        fclose(f); rename(tmp,ckpt); }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double sums = (double)count*(double)(1ULL<<chunkLog);
    fprintf(stderr,"enumerated %.6g sign vectors in %.1f s  ->  %.4f Gsums/s\n",
            sums, el, sums/el*1e-9);

    if (part) write_part(part, N, chunkLog, from, count, total, el);

    if (from == 0 && count == totalChunks) return finish(N, total);
    printf("PARTIAL n=%d chunklog=%d chunks %lld..%lld  sum = ", N, chunkLog, from, from+count-1);
    print_i160(total); printf("\n");
    return 0;
}
