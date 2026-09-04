/* Is there an edge weighting w with  cr(M) + sum_{e in M} w_e = const (mod 2)
 * over ALL Langford matchings M?  If yes, the Pfaffian counts them WITHOUT the
 * sign problem, and Godfrey's 4^n collapses to 2^n * poly.
 * Pure GF(2) consistency test, solved incrementally while enumerating.        */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int N, M2, NE, NW;          /* NE = #edges, NW = words per row       */
static int eoff[64];               /* edge index base for each difference   */
static uint64_t *basis;            /* basis[pivot*NW]  (0 = unused)         */
static int *has;                   /* pivot occupied?                       */
static long long nmatch=0, nred=0, bad=0;
static uint64_t occ, rights;       /* occupied positions / right endpoints  */
static uint64_t used;              /* used differences                      */
static uint64_t *row;              /* current partial row                   */
static int cr;

static inline void setbit(uint64_t*v,int i){ v[i>>6] ^= 1ULL<<(i&63); }

/* add the augmented row to the system; detect inconsistency */
static void feed(void)
{
    static uint64_t r[64];
    memcpy(r, row, NW*8);
    setbit(r, NE);                       /* the constant c */
    if (cr & 1) setbit(r, NE+1);         /* rhs            */
    nmatch++;
    for (int p = 0; p <= NE; p++) {      /* pivots over LHS columns only */
        if (!((r[p>>6]>>(p&63))&1)) continue;
        if (has[p]) { for(int k=0;k<NW;k++) r[k]^=basis[(size_t)p*NW+k]; }
        else { memcpy(basis+(size_t)p*NW, r, NW*8); has[p]=1; nred++; return; }
    }
    /* LHS fully reduced to zero: consistent only if rhs is zero too */
    if ((r[(NE+1)>>6]>>((NE+1)&63))&1) bad++;
}

static void rec(void)
{
    int p = 0; while (p < M2 && ((occ>>p)&1)) p++;
    if (p == M2) { feed(); return; }
    for (int d = 2; d <= N+1; d++) {
        if ((used>>d)&1) continue;
        int q = p + d; if (q >= M2) continue;
        if ((occ>>q)&1) continue;
        /* crossings with earlier chords: their right end strictly inside (p,q) */
        uint64_t span = (q-p-1>=64)?~0ULL:(((1ULL<<(q-p-1))-1)<<(p+1));
        int add = __builtin_popcountll(rights & span);
        int ei = eoff[d] + p;
        setbit(row, ei); used|=1ULL<<d; occ|=(1ULL<<p)|(1ULL<<q);
        rights|=1ULL<<q; cr+=add;
        rec();
        cr-=add; rights&=~(1ULL<<q); occ&=~((1ULL<<p)|(1ULL<<q));
        used&=~(1ULL<<d); setbit(row, ei);
    }
}

int main(int argc,char**argv)
{
    N=atoi(argv[1]); M2=2*N;
    NE=0; for(int d=2;d<=N+1;d++){ eoff[d]=NE; NE += M2-d; }
    NW = (NE+2+63)/64;
    basis=calloc((size_t)(NE+1)*NW,8); has=calloc(NE+1,4); row=calloc(NW,8);
    occ=rights=used=0; cr=0;
    rec();
    printf("n=%2d  edges=%4d  matchings=%lld  rank=%lld  inconsistent rows=%lld  -> %s\n",
           N,NE,nmatch,nred,bad, bad? "NO Pfaffian weighting":"*** WEIGHTING EXISTS ***");
    return 0;
}
