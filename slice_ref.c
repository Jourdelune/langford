/* slice_ref.c -- recalcule une tranche `PART=` a partir de la DEFINITION.
 *
 * Le noyau GPU passe par : decomposition de parite, tables SWAR, arbre de
 * produit 160 bits en chaines de retenue PTX, chemin rapide sur la magnitude.
 * Ce programme ne partage RIEN de cela.  Pour chaque point enumere il
 * reconstruit la suite X entiere sur 2n positions, calcule chaque A_i par un
 * popcount direct sur la definition A_i = sum_k x_k x_{k+i}, et multiplie les
 * n facteurs dans un accumulateur 160 bits ecrit a la main.
 *
 * Si les deux `PART=` coincident au bit pres, alors sur cette tranche :
 *   - la decomposition de parite est juste (elle n'est pas utilisee ici) ;
 *   - l'arithmetique 160 bits du noyau est juste (celle-ci est independante) ;
 *   - le chemin rapide de l'arbre de produit n'a rien casse ;
 *   - le predicat canonique selectionne les memes points.
 *
 * usage : ./slice_ref <n> <vhi> [K]      (K = 7 par defaut, comme le noyau)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef struct { uint32_t w[5]; } u160;                 /* complement a deux */

static void mul_small(u160 *z, uint32_t a){
    uint64_t c = 0;
    for (int k = 0; k < 5; k++){ uint64_t t = (uint64_t)z->w[k]*a + c; z->w[k] = (uint32_t)t; c = t >> 32; }
}
static void add160(u160 *z, const u160 *x){
    uint64_t c = 0;
    for (int k = 0; k < 5; k++){ uint64_t t = (uint64_t)z->w[k] + x->w[k] + c; z->w[k] = (uint32_t)t; c = t >> 32; }
}
static void sub160(u160 *z, const u160 *x){
    uint64_t b = 0;
    for (int k = 0; k < 5; k++){ uint64_t t = (uint64_t)z->w[k] - x->w[k] - b;
        z->w[k] = (uint32_t)t; b = (t >> 63) & 1ULL; }
}
static void shl1(u160 *z){                              /* le x2 du noyau principal */
    uint32_t c = 0;
    for (int k = 0; k < 5; k++){ uint32_t v = z->w[k]; z->w[k] = (v << 1) | c; c = v >> 31; }
}

static int N, M;
static uint32_t brev32(uint32_t x){
    x = ((x>>1)&0x55555555u)|((x&0x55555555u)<<1); x = ((x>>2)&0x33333333u)|((x&0x33333333u)<<2);
    x = ((x>>4)&0x0F0F0F0Fu)|((x&0x0F0F0F0Fu)<<4); x = ((x>>8)&0x00FF00FFu)|((x&0x00FF00FFu)<<8);
    return (x>>16)|(x<<16);
}

/* A_i directement sur la suite complete : la definition, rien d'autre. */
static int Adirect(uint64_t b, int i){
    int w = M - i;
    uint64_t mk = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
    return w - 2*__builtin_popcountll((b ^ (b >> i)) & mk);
}

int main(int argc, char **argv){
    if (argc < 3){ fprintf(stderr, "usage: %s <n> <vhi> [K]\n", argv[0]); return 1; }
    N = atoi(argv[1]); M = 2*N;
    const long long VHI = atoll(argv[2]);
    const int K = argc > 3 ? atoi(argv[3]) : 7;
    const int FREE = N - 1;
    const uint32_t ALLB = (1u << N) - 1u;
    const uint32_t NL = 1u << K;
    const long long b0 = ((long long)VHI << K) >> 8;    /* premier bloc lance */
    const uint32_t u0 = (uint32_t)(b0 < 0 ? 0 : b0) * 256u;
    const uint32_t uend = 1u << FREE;

    u160 acc; memset(&acc, 0, sizeof acc);
    for (uint32_t u = u0; u < uend; u++){
        uint32_t rvF = brev32(u << 1) >> (32 - N);
        uint32_t o = (rvF & 1u) ? (rvF ^ ALLB) : rvF;
        for (uint32_t el = 0; el < NL; el++){
            uint32_t v = ((uint32_t)VHI << K) | el;
            if (!(v < u)) continue;                     /* predicat canonique du noyau */
            uint32_t e = v << 1;
            /* reconstruire X : rangee o aux positions impaires, e aux paires */
            uint64_t b = 0;
            for (int j = 0; j < N; j++){
                if ((o >> j) & 1u) b |= 1ULL << (2*j);
                if ((e >> j) & 1u) b |= 1ULL << (2*j + 1);
            }
            u160 z; memset(&z, 0, sizeof z); z.w[0] = 1;
            int neg = 0, zero = 0;
            for (int i = 2; i <= N + 1; i++){
                int a = Adirect(b, i);
                if (!a){ zero = 1; break; }
                if (a < 0){ neg ^= 1; a = -a; }
                mul_small(&z, (uint32_t)a);
            }
            if (zero) continue;
            if (__builtin_popcountll(b) & 1) neg ^= 1;  /* le poids prod x_k */
            if (neg) sub160(&acc, &z); else add160(&acc, &z);
        }
    }
    shl1(&acc);                                          /* poids 2 : moitie canonique */
    printf("PART=%08x:%08x:%08x:%08x:%08x   (ref n=%d vhi=%lld)\n",
           acc.w[4], acc.w[3], acc.w[2], acc.w[1], acc.w[0], N, VHI);
    return 0;
}
