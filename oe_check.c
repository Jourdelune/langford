/* Verifie la decomposition de parite:  b <-> (o,e), o = positions impaires,
 * e = positions paires.  Ecart pair 2m : A = P_m(o) + Q_m(e)  (SEPARES).
 * Ecart impair 2m+1 : bilineaire en (o,e).                                  */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int N,M;
static int Adirect(uint64_t b,int i){ int w=M-i; uint64_t mk=(w>=64)?~0ULL:((1ULL<<w)-1);
    return w-2*__builtin_popcountll((b^(b>>i))&mk); }
static int P_(uint32_t o,int m){ int L=N-m; return L-2*__builtin_popcount((o^(o>>m))&((1u<<L)-1)); }
static int cross(uint32_t o,uint32_t e,int m){          /* sum O_j E_{j+m} */
    int L=N-m; return L-2*__builtin_popcount((o^(e>>m))&((1u<<L)-1)); }
static int cross2(uint32_t o,uint32_t e,int m){         /* sum E_j O_{j+m+1} */
    int L=N-m-1; if(L<=0) return 0;
    return L-2*__builtin_popcount((e^(o>>(m+1)))&((1u<<L)-1)); }
int main(int argc,char**argv){
    N=atoi(argv[1]); M=2*N; uint64_t st=12345;
    long long bad=0,tested=0;
    for(int t=0;t<200000;t++){
        uint64_t b=0; for(int w=0;w<2;w++){ st^=st<<13;st^=st>>7;st^=st<<17; b=(b<<32)^(st&0xffffffffu);}
        b &= (M>=64)?~0ULL:((1ULL<<M)-1);
        uint32_t o=0,e=0;
        for(int j=0;j<N;j++){ if((b>>(2*j))&1) o|=1u<<j; if((b>>(2*j+1))&1) e|=1u<<j; }
        for(int i=2;i<=N+1;i++){
            int a=Adirect(b,i), c;
            if(i%2==0){ int m=i/2; c=P_(o,m)+P_(e,m); }
            else      { int m=(i-1)/2; c=cross(o,e,m)+cross2(o,e,m); }
            tested++; if(a!=c){ if(bad<3) printf("  MISMATCH n=%d i=%d direct=%d decomp=%d\n",N,i,a,c); bad++; }
        }
    }
    printf("n=%2d  %lld lags tested, %lld mismatches  -> %s\n",N,tested,bad,bad?"BROKEN":"OK");
    return 0;
}
