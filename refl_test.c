/* Derive and validate the reflection symmetry sigma = canon(reverse).
 *
 * Canonical set: b_{M-1} = b_{M-2} = 0 (the Z2xZ2 subgroup {0,ALT,~ALT,FULL}
 * acts simply transitively on those two bits).  sigma(b) = the unique canonical
 * representative of reverse(b).  If sigma is an involution and preserves the
 * summand, half the canonical set suffices -> another factor 2.
 *
 * Also validates the concrete enumeration used by the GPU:
 *   class A = (b0,b1)=(0,1) enumerated fully, weight 2
 *   classes B=(0,0), C=(1,1): enumerate H <= W only, weight 2 (diagonal weight 1)
 * where HI = bits [h..M-3], LO = bits [2..h-1], W = image of LO under sigma.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int n, M;
static uint64_t FULLM, ALT;

static uint64_t revbits(uint64_t b){            /* p -> M-1-p */
    uint64_t r = 0;
    for (int p = 0; p < M; p++) if ((b>>p)&1) r |= 1ULL << (M-1-p);
    return r;
}
static uint64_t canon(uint64_t c){              /* force bits M-1, M-2 to zero */
    uint64_t g[4] = {0, ALT, ALT ^ FULLM, FULLM};
    for (int k = 0; k < 4; k++){
        uint64_t d = c ^ g[k];
        if (!((d >> (M-1)) & 1) && !((d >> (M-2)) & 1)) return d;
    }
    fprintf(stderr,"canon failed\n"); exit(2);
}
static uint64_t sigma(uint64_t b){ return canon(revbits(b)); }

static __int128 term(uint64_t b){
    __int128 p = 1;
    for (int i = 2; i <= n+1; i++){
        int w = M - i;
        uint64_t mk = (1ULL << w) - 1;
        p *= (__int128)(w - 2*__builtin_popcountll((b ^ (b>>i)) & mk));
        if (!p) break;
    }
    return (__builtin_popcountll(b) & 1) ? -p : p;
}

int main(int argc, char **argv){
    if(argc < 2){ fprintf(stderr,"usage: %s <n>\n", argv[0]); return 1; }
    n = atoi(argv[1]); M = 2*n;
    if(n < 3 || n > 16){ fprintf(stderr,"n hors domaine (3..16)\n"); return 1; }
    FULLM = (M >= 64) ? ~0ULL : ((1ULL<<M)-1);
    ALT = 0; for (int p = 1; p < M; p += 2) ALT |= 1ULL<<p;

    int Lf = M-2;                 /* free bits 0..M-3 */
    int h  = 2 + (Lf-2)/2;        /* HI starts here; LO = [2,h), HI = [h,M-2) */
    uint64_t nfree = 1ULL << Lf;

    /* --- 1. sigma is an involution and preserves the summand --- */
    long long bad_inv = 0, bad_term = 0, fixed = 0;
    __int128 ref = 0;
    for (uint64_t b = 0; b < nfree; b++){
        uint64_t s = sigma(b);
        if (sigma(s) != b) bad_inv++;
        if (term(s) != term(b)) bad_term++;
        if (s == b) fixed++;
        ref += term(b);
    }
    printf("n=%2d  M=%2d  canonical set 2^%d\n", n, M, Lf);
    printf("  involution failures : %lld\n", bad_inv);
    printf("  summand mismatches  : %lld\n", bad_term);
    printf("  fixed points        : %lld\n", fixed);

    /* --- 2. sigma acts on (b0,b1) by swapping them --- */
    long long bad_cls = 0;
    for (uint64_t b = 0; b < nfree; b++){
        uint64_t s = sigma(b);
        if ((((s>>0)&1) != ((b>>1)&1)) || (((s>>1)&1) != ((b>>0)&1))) bad_cls++;
    }
    printf("  class map (b0,b1)->(b1,b0) failures: %lld\n", bad_cls);

    /* --- 3. the half-enumeration reproduces the full sum --- */
    int nHI = M-2-h, nLO = h-2;
    if (nHI != nLO){ printf("  (HI=%d LO=%d bits: split not balanced)\n", nHI, nLO); }
    __int128 half = 0;
    for (uint64_t b = 0; b < nfree; b++){
        int b0 = (b>>0)&1, b1 = (b>>1)&1;
        if (b0 == 0 && b1 == 1){ half += 2*term(b); continue; }   /* class A, weight 2 */
        if (b0 != b1) continue;                                    /* class (1,0): skipped */
        uint64_t s = sigma(b);
        uint64_t H  = (b>>h) & ((1ULL<<nHI)-1);
        uint64_t W  = (s>>h) & ((1ULL<<nHI)-1);   /* W = image of LO(b) */
        if (H < W) half += 2*term(b);
        else if (H == W) half += term(b);
    }
    printf("  full sum   = %lld...\n  half*sym   = %lld...   %s\n",
           (long long)(ref & 0x7fffffffffff), (long long)(half & 0x7fffffffffff),
           (ref == half) ? "*** HALF-ENUMERATION EXACT ***" : "MISMATCH");
    return 0;
}
