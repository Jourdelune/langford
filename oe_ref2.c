/* Reference CPU dans les coordonnees (o,e).
 * Le groupe de Klein {id, nu, eps, nu.eps} devient, en coordonnees de parite,
 * "negation independante de chaque rangee" -- il agit simplement transitivement
 * sur (o_1, e_1), donc on epingle o_1 = e_1 = 0 et on multiplie par 4.        */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int N;
static inline int P_(uint32_t o,int m){ int L=N-m; return L-2*__builtin_popcount((o^(o>>m))&((1u<<L)-1)); }
static inline int X1(uint32_t o,uint32_t e,int m){ int L=N-m;
    return L-2*__builtin_popcount((o^(e>>m))&((1u<<L)-1)); }
static inline int X2(uint32_t o,uint32_t e,int m){ int L=N-m-1; if(L<=0) return 0;
    return L-2*__builtin_popcount((e^(o>>(m+1)))&((1u<<L)-1)); }
static void print128(__int128 v){ if(v<0){putchar('-');v=-v;} char s[48]; int n=0;
    if(!v){putchar('0');return;} while(v){ s[n++]='0'+(int)(v%10); v/=10; }
    while(n--) putchar(s[n]); }
int main(int argc,char**argv){
    N=atoi(argv[1]); int M=2*N;
    uint32_t H=1u<<(N-1);                       /* o = 2u, e = 2v : bit 0 nul */
    /* reflexion : sigma(o,e) = (f(e), f(o)) avec f = canon o rev.
     * Le representant canonique est e <= f(o) ; f(o) est constant a o fixe. */
    uint32_t ALLB=(N>=32)?0xFFFFFFFFu:((1u<<N)-1u);
    __int128 tot=0;
    for(uint32_t u=0;u<H;u++){ uint32_t o=u<<1;
        uint32_t rv;
        { uint32_t x=o,y=0; for(int j=0;j<N;j++) if((x>>j)&1) y|=1u<<(N-1-j); rv=y; }
        uint32_t F=(rv&1u)?(rv^ALLB):rv;
        for(uint32_t v=0;v<H;v++){ uint32_t e=v<<1;
            int wgt = (e<F)?2:((e==F)?1:0);
            if(!wgt) continue;
            __int128 p=1; int zero=0;
            for(int m=1;2*m<=N+1;m++){ int a=P_(o,m)+P_(e,m); if(!a){zero=1;break;} p*=a; }
            if(zero) continue;
            for(int m=1;2*m+1<=N+1;m++) p*=(X1(o,e,m)+X2(o,e,m));
            p*=wgt;
            if((__builtin_popcount(o)+__builtin_popcount(e))&1) tot-=p; else tot+=p;
        } }
    tot*=4;
    /* tot doit valoir 2^{2N} * V(N) */
    for(int b=0;b<M;b++) if(tot & (((__int128)1)<<b)){ printf("n=%d  NOT divisible by 2^%d\n",N,M); return 1; }
    __int128 V=tot>>M;
    printf("n=%2d   V = 2*L = ",N); print128(V); printf("   L = "); print128(V/2); printf("\n");
    return 0;
}
