/* find_langford.c -- exhibit actual Langford sequences L(2,n).
 *
 * This is the EASY half of the problem: deciding existence and producing
 * solutions is polynomial-time in practice (n = 0 or 3 mod 4 is necessary and
 * sufficient, and explicit constructions are known).  Only *counting* all of
 * them is hard.  Bitmask backtracking, values placed largest-first.
 *
 *   usage: ./find_langford n [howmany]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int n, M, want, found;
static uint64_t occ;
static int seq[128];

static void emit(void){
    /* independent verification before printing */
    int pos[128][2], cnt[128];
    for (int v = 1; v <= n; v++) cnt[v] = 0;
    for (int p = 0; p < M; p++){
        int v = seq[p];
        if (v < 1 || v > n) { fprintf(stderr,"BAD: slot %d empty\n",p); exit(2); }
        if (cnt[v] >= 2) { fprintf(stderr,"BAD: value %d used %d times\n",v,cnt[v]+1); exit(2); }
        pos[v][cnt[v]++] = p;
    }
    for (int v = 1; v <= n; v++){
        if (cnt[v] != 2) { fprintf(stderr,"BAD: value %d appears %d times\n",v,cnt[v]); exit(2); }
        if (pos[v][1] - pos[v][0] != v + 1) {
            fprintf(stderr,"BAD: value %d gap %d != %d\n",v,pos[v][1]-pos[v][0]-1,v); exit(2); }
    }
    printf("%2d:", ++found);
    for (int p = 0; p < M; p++) printf(" %d", seq[p]);
    printf("\n");
}

static void place(int v){
    if (found >= want) return;
    if (v == 0){ emit(); return; }
    for (int p = 0; p + v + 1 < M; p++){
        uint64_t m = (1ULL << p) | (1ULL << (p + v + 1));
        if (occ & m) continue;
        occ |= m; seq[p] = seq[p+v+1] = v;
        place(v - 1);
        occ &= ~m;
        if (found >= want) return;
    }
}

int main(int argc, char **argv){
    n = argc > 1 ? atoi(argv[1]) : 31;
    want = argc > 2 ? atoi(argv[2]) : 1;
    M = 2*n;
    if (M > 64){ fprintf(stderr,"n>32 not supported by the 64-bit mask\n"); return 1; }
    if (n % 4 != 0 && n % 4 != 3){
        printf("L(2,%d) has no solution (n must be 0 or 3 mod 4)\n", n); return 0;
    }
    place(n);
    if (!found) printf("no solution found\n");
    else fprintf(stderr,"%d verified solution(s) for n=%d\n", found, n);
    return 0;
}
