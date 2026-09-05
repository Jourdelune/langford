/* Verifie exactement la decomposition v7 des ecarts IMPAIRS :
 *   A_{2m+1}(o,e) = Base_m(o,e_hi) + delta_m(o_fix,e_lo) + 4*sum(o_a & e_c)
 * ou o_var = bits [N-9,N-2] (les seuls bits de o portes par threadIdx),
 * e_lo = bits 1..K.  Aucun terme ne doit manquer. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int popc(unsigned x){int c=0;while(x){c+=x&1;x>>=1;}return c;}

/* A_{2m+1} par la formule du noyau (deux correlations) */
static int gap_odd(int N,int m,unsigned o,unsigned e){
    int L1=N-m, L2=N-m-1, A=0;
    A += L1 - 2*popc((o ^ (e>>m)) & ((1u<<L1)-1u));
    if(L2>0) A += L2 - 2*popc((e ^ (o>>(m+1))) & ((1u<<L2)-1u));
    return A;
}
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):31, K=argc>2?atoi(argv[2]):7;
    int MO=N/2, A0=N-9, A1=N-2, bad=0; long tested=0;
    unsigned seed=12345;
    for(int trial=0; trial<200000; trial++){
        seed=seed*1664525u+1013904223u; unsigned r1=seed;
        seed=seed*1664525u+1013904223u; unsigned r2=seed;
        seed=seed*1664525u+1013904223u; unsigned r3=seed;
        unsigned o=(r1^(r2<<11))&((1u<<N)-1u); o&=~1u;          /* o_1 = 0 */
        unsigned ehi=(r2&((1u<<N)-1u)) & ~((1u<<(K+1))-1u);     /* bits K+1.. */
        unsigned el=r3&((1u<<K)-1u);
        unsigned e=ehi|(el<<1);                                  /* e_1 = 0 */
        unsigned e0=ehi;                                         /* e_lo = 0 */
        for(int m=1;m<=MO;m++){
            int Base=gap_odd(N,m,o,e0);
            /* delta : ne lit que les bits de o HORS o_var */
            int d=0;
            for(int c=1;c<=K;c++){
                int ec=(int)((e>>c)&1u);
                if(!ec) continue;
                if(c>=m){ int a=c-m; d += -2*(1-2*(int)((o>>a)&1u)); }
                if(c<=N-m-2){ int a=c+m+1;
                    if(a>=A0&&a<=A1) d += -2;                    /* reporte au drain */
                    else             d += -2*(1-2*(int)((o>>a)&1u)); }
            }
            /* correction portee par le drain */
            int t=0;
            for(int c=1;c<=K;c++)
                if(c<=N-m-2 && c+m+1>=A0 && c+m+1<=A1)
                    t += (int)(((o>>(c+m+1)) & (e>>c)) & 1u);
            int got=Base+d+4*t, want=gap_odd(N,m,o,e);
            tested++;
            if(got!=want){ if(bad<5) printf("DIVERGENCE N=%d m=%d o=%08x e=%08x  got=%d want=%d\n",N,m,o,e,got,want); bad++; }
        }
    }
    printf("N=%2d K=%d : %ld ecarts testes, %d divergences  (o_var=bits %d..%d)\n",N,K,tested,bad,A0,A1);
    /* combien de termes croises restent au drain ? */
    int nx=0; for(int m=1;m<=MO;m++) for(int c=1;c<=K;c++)
        if(c<=N-m-2 && c+m+1>=A0 && c+m+1<=A1){ nx++; printf("   terme croise : m=%d c=%d (o bit %d)\n",m,c,c+m+1); }
    printf("   total %d termes croises\n",nx);
    return bad!=0;
}
