/* Is the order-8 group really all of it?  Measure the fibres of
 *    b  |-->  (A_2(b), ..., A_{n+1}(b))
 * The known stabiliser of this tuple is {id, nu, rho, nu.rho} = 4.
 * Anything bigger means unexploited symmetry -> fewer evaluation points.  */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static int cmp(const void*a,const void*b){ uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b;
    return x<y?-1:x>y?1:0; }
int main(int argc,char**argv){
    int N=atoi(argv[1]), M=2*N; uint64_t T=1ULL<<M;
    uint64_t *h=malloc(T*8); if(!h){puts("oom");return 1;}
    for(uint64_t b=0;b<T;b++){
        uint64_t k=1469598103934665603ULL;                 /* FNV-1a over the tuple */
        for(int i=2;i<=N+1;i++){ int w=M-i; uint64_t mk=(1ULL<<w)-1;
            int P=w-2*__builtin_popcountll((b^(b>>i))&mk);
            k=(k^(uint64_t)(P+128))*1099511628211ULL; }
        h[b]=k;
    }
    qsort(h,T,8,cmp);
    long long hist[65]={0}; uint64_t i=0; long long maxf=0;
    while(i<T){ uint64_t j=i; while(j<T&&h[j]==h[i]) j++;
        long long f=j-i; if(f<65) hist[f]++; if(f>maxf) maxf=f; i=j; }
    printf("n=%2d  points=%llu  fibre sizes:",N,(unsigned long long)T);
    for(int f=1;f<=16;f++) if(hist[f]) printf("  %d:%lld",f,hist[f]);
    printf("   max=%lld\n",maxf);
    free(h); return 0;
}
