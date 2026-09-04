/* Is the crossing parity of a Langford pairing an invariant?
 * If yes,  #solutions = |Pfaffian sum|, computable in 2^n * n^3.
 * Chords: value v occupies (p, p+v+1).  cr(M) = # pairs of crossing chords. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int n, M; static uint64_t occ; static int a[64], b[64];
static long long cnt[2];
static void rec(int v){
    if (v == 0){
        int cr = 0;
        for (int i = 1; i <= n; i++) for (int j = i+1; j <= n; j++)
            if ((a[i] < a[j] && a[j] < b[i] && b[i] < b[j]) ||
                (a[j] < a[i] && a[i] < b[j] && b[j] < b[i])) cr++;
        cnt[cr & 1]++; return;
    }
    for (int p = 0; p + v + 1 < M; p++){
        uint64_t m = (1ULL<<p) | (1ULL<<(p+v+1));
        if (occ & m) continue;
        occ |= m; a[v] = p; b[v] = p+v+1;
        rec(v-1);
        occ &= ~m;
    }
}
int main(int argc, char**argv){
    n = atoi(argv[1]); M = 2*n; rec(n);
    printf("n=%2d  total=%lld   even-crossing=%lld  odd-crossing=%lld   %s\n",
           n, cnt[0]+cnt[1], cnt[0], cnt[1],
           (cnt[0]==0||cnt[1]==0) ? "*** PARITY IS CONSTANT ***" : "parity is NOT constant");
    return 0;
}
