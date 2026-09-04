/* Same question, but with roots of unity instead of +-1.
 * Kasteleyn only needs  sign(M) * prod_{e in M} w_e = const  for Langford M.
 * With w_e = zeta^{t_e}, zeta = exp(2*pi*i/2^k), that is
 *      sum_{e in M} t_e + 2^{k-1} * cr(M) = C   (mod 2^k)
 * a linear system over Z/2^k -- strictly weaker than the GF(2) one (k=1).   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int N,M2,NE,NC,KB;  static uint32_t MOD_MASK, HALF;
static uint32_t *basis; static int *hasp;
static long long nmatch=0,bad=0; static int rank=0;
static uint64_t occ,rights,used; static uint32_t *row; static int cr;

static uint32_t inv_odd(uint32_t a){ uint32_t x=a; for(int i=0;i<5;i++) x*=2-a*x; return x; }
static int val2(uint32_t a){ return a? __builtin_ctz(a) : 32; }
static inline uint32_t msk(uint32_t v){ return KB==32? v : (v & MOD_MASK); }

static void feed(void)
{
    static uint32_t r[512]; nmatch++;
    memcpy(r,row,(size_t)NC*4);
    r[NE]=1;                                   /* the constant C */
    r[NC-1]= msk((cr&1)? HALF : 0);            /* rhs            */
    for(int j=0;j<=NE;j++){
        if(!r[j]) continue;
        if(!hasp[j]){ memcpy(basis+(size_t)j*NC,r,(size_t)NC*4); hasp[j]=1; rank++; return; }
        uint32_t *p=basis+(size_t)j*NC;
        int v=val2(p[j]), u=val2(r[j]);
        if(u<v){ for(int t=0;t<NC;t++){ uint32_t tmp=p[t]; p[t]=r[t]; r[t]=tmp; } int s=v; v=u; u=s; }
        uint32_t q = msk(((r[j]>>v) * inv_odd(p[j]>>v)));
        for(int t=0;t<NC;t++) r[t]=msk(r[t]-q*p[t]);
    }
    if(r[NC-1]) bad++;                          /* 0 = nonzero  -> impossible */
}

static void rec(void){
    int p=0; while(p<M2 && ((occ>>p)&1)) p++;
    if(p==M2){ feed(); return; }
    for(int d=2;d<=N+1;d++){
        if((used>>d)&1) continue;
        int q=p+d; if(q>=M2||((occ>>q)&1)) continue;
        uint64_t span=(q-p-1>=64)?~0ULL:(((1ULL<<(q-p-1))-1)<<(p+1));
        int add=__builtin_popcountll(rights&span);
        int ei=0; for(int dd=2;dd<d;dd++) ei+=M2-dd; ei+=p;
        row[ei]=msk(row[ei]+1); used|=1ULL<<d; occ|=(1ULL<<p)|(1ULL<<q);
        rights|=1ULL<<q; cr+=add; rec(); cr-=add;
        rights&=~(1ULL<<q); occ&=~((1ULL<<p)|(1ULL<<q)); used&=~(1ULL<<d);
        row[ei]=msk(row[ei]-1);
    }
}

int main(int argc,char**argv){
    N=atoi(argv[1]); KB=atoi(argv[2]); M2=2*N;
    MOD_MASK = (KB==32)?0xFFFFFFFFu:((1u<<KB)-1u); HALF = 1u<<(KB-1);
    NE=0; for(int d=2;d<=N+1;d++) NE+=M2-d; NC=NE+2;
    basis=calloc((size_t)(NE+1)*NC,4); hasp=calloc(NE+1,4); row=calloc(NC,4);
    rec();
    printf("n=%2d  Z/2^%-2d  matchings=%lld  rank=%d  impossible rows=%lld  -> %s\n",
           N,KB,nmatch,rank,bad, bad?"no weighting":"*** WEIGHTING EXISTS ***");
    return 0;
}
