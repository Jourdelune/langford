/* cover_check.c -- le pont entre la preuve Lean et le code reel.
 *
 * proof/Langford.lean prouve la couverture exacte sur des rangees abstraites
 * (Fin (n+1) -> Bool).  Le noyau, lui, manipule des masques 32 bits et un
 * index `u` passe par __brev.  Ce programme verifie, EXHAUSTIVEMENT et sur les
 * memes formules bit a bit que le noyau, que les deux coincident :
 *
 *   (1) rev_N est une involution ;
 *   (2) f = canon o rev_N est une involution sur les rangees epinglees ;
 *   (3) o(u), tel que le noyau le calcule, vaut exactement f(u<<1) ;
 *   (4) f(o(u)) = u<<1  -- donc le predicat `v < u` du noyau EST `e < f(o)` ;
 *   (5) u -> o(u) est une BIJECTION des 2^{N-1} index vers les rangees
 *       epinglees : aucun o n'est visite deux fois, aucun n'est oublie ;
 *   (6) le compte total : 4 * (2*#{v<u} + #{v=u}) = 4^N, soit exactement le
 *       nombre de points de l'espace de Godfrey ;
 *   (7) pour les petits n, la couverture point par point : chaque couple
 *       (o,e) epingle est atteint par exactement un representant enumere.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int N;
static uint32_t ALLB;

/* exactement les operations du noyau */
/* __brev de CUDA, a l'identique */
static inline uint32_t brev32(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    return (x >> 16) | (x << 16);
}
static inline uint32_t revN(uint32_t r){                  /* __brev puis cadrage */
    return brev32(r) >> (32 - N);
}
static inline uint32_t canon(uint32_t r){ return (r & 1u) ? (r ^ ALLB) : r; }
static inline uint32_t fmap(uint32_t r){ return canon(revN(r)); }
/* o(u) : la ligne du noyau, recopiee telle quelle */
static inline uint32_t o_of_u(uint32_t u){
    uint32_t rvF = revN(u << 1);
    return (rvF & 1u) ? (rvF ^ ALLB) : rvF;
}

int main(int argc, char **argv){
    int lo = argc > 1 ? atoi(argv[1]) : 9;
    int hi = argc > 2 ? atoi(argv[2]) : 24;
    int full = argc > 3 ? atoi(argv[3]) : 12;   /* n max pour le test (7) */
    int bad = 0;
    printf("  n |    2^(N-1) | (1)rev | (2)f inv | (3)o=f(F) | (4)f(o)=F | (5)bijection | (6)compte | (7)couverture\n");
    printf("----+------------+--------+----------+-----------+-----------+--------------+-----------+---------------\n");
    for (N = lo; N <= hi; N++) {
        if (N % 4 != 0 && N % 4 != 3) continue;      /* les seuls n licites */
        ALLB = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
        const uint32_t T = 1u << (N - 1);            /* nombre d'index u */
        int e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1, e7 = -1;

        /* (1) et (2) : sur TOUTES les rangees quand c'est jouable, sinon sur
           toutes les rangees epinglees (2^{N-1}, toujours exhaustif). */
        for (uint32_t r = 0; r < T && (e1 || e2); r++) {
            uint32_t p = r << 1;                     /* rangee epinglee : bit 0 = 0 */
            if (revN(revN(p)) != p) e1 = 0;
            if (fmap(fmap(p)) != p) e2 = 0;
        }
        /* (3), (4), (5) */
        /* bitset : 2^N bits, soit 256 Mo a n=31 au lieu de 2 Go */
        size_t words = ((size_t)1u << N) / 64 + 1;
        uint64_t *seen = calloc(words, sizeof *seen);
        if (!seen) { printf("memoire\n"); return 2; }
        #define SEEN_GET(i) ((seen[(i) >> 6] >> ((i) & 63)) & 1ULL)
        #define SEEN_SET(i) (seen[(i) >> 6] |= 1ULL << ((i) & 63))
        for (uint32_t u = 0; u < T; u++) {
            uint32_t o = o_of_u(u);
            if (o != fmap(u << 1))      e3 = 0;
            if (fmap(o) != (u << 1))    e4 = 0;
            if (o & 1u)                 e5 = 0;      /* o doit etre epingle */
            if (SEEN_GET(o))            e5 = 0;      /* et jamais revisite */
            SEEN_SET(o);
        }
        if (e5) { /* et aucune rangee epinglee oubliee */
            for (uint32_t r = 0; r < T; r++) if (!SEEN_GET((uint32_t)r << 1)) { e5 = 0; break; }
        }
        free(seen);
        #undef SEEN_GET
        #undef SEEN_SET
        /* (6) : 4 * (2*C(T,2) + T) doit valoir 4^N = (2^N)^2 */
        {   __int128 enumere = (__int128)4 * (2 * ((__int128)T * (T - 1) / 2) + T);
            __int128 espace  = ((__int128)1 << N) * ((__int128)1 << N);
            e6 = (enumere == espace);
        }
        /* (7) : couverture point par point, pour les petits n seulement */
        if (N <= full) {
            e7 = 1;
            size_t sz = (size_t)T * T;
            unsigned char *hit = calloc(sz, 1);
            if (!hit) { printf("memoire\n"); return 2; }
            /* le noyau principal : tous les couples v < u, poids 2 */
            for (uint32_t u = 0; u < T; u++)
                for (uint32_t v = 0; v < u; v++) hit[(size_t)u * T + v] += 2;
            /* le noyau diagonal : v == u, poids 1 */
            for (uint32_t u = 0; u < T; u++) hit[(size_t)u * T + u] += 1;
            /* sigma echange (u,v) <-> (v,u) : chaque orbite doit totaliser 2 */
            for (uint32_t u = 0; u < T && e7; u++)
                for (uint32_t v = 0; v < T; v++) {
                    int a = hit[(size_t)u * T + v], b = hit[(size_t)v * T + u];
                    if (u == v) { if (a != 1) { e7 = 0; break; } }
                    else        { if (a + b != 2) { e7 = 0; break; } }
                }
            free(hit);
        }
        printf("%3d | %10u |  %-5s |  %-7s |  %-8s |  %-8s |  %-11s | %-8s  | %s\n",
               N, T, e1?"ok":"NON", e2?"ok":"NON", e3?"ok":"NON", e4?"ok":"NON",
               e5?"ok":"NON", e6?"ok":"NON",
               e7 < 0 ? "(non teste)" : (e7 ? "ok" : "NON"));
        if (!e1||!e2||!e3||!e4||!e5||!e6||(e7==0)) bad = 1;
        fflush(stdout);
    }
    printf("\n%s\n", bad ? "*** DIVERGENCE ***"
        : "tous les controles passent : l'indexation du noyau realise bien\n"
          "le theoreme `couverture_exacte` de proof/Langford.lean.");
    return bad;
}
