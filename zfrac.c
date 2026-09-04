#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
/* fraction of sign vectors b in {0,1}^{2n} for which some A_i(b)=0  */
int main(int argc,char**argv){
    int N=atoi(argv[1]); long long S=atoll(argv[2]);
    int M=2*N; uint64_t st=88172645463325252ULL;
    long long live=0; long long hist[40]={0};
    for(long long s=0;s<S;s++){
        uint64_t b=0;
        for(int w=0;w<2;w++){ st^=st<<13; st^=st>>7; st^=st<<17; b=(b<<32)^(st&0xffffffffu);}    
        b &= (M>=64)?~0ULL:((1ULL<<M)-1);
        int nz=0,zc=0;
        for(int i=2;i<=N+1;i++){
            int w=M-i; uint64_t mk=(w>=64)?~0ULL:((1ULL<<w)-1);
            int P=w-2*__builtin_popcountll((b^(b>>i))&mk);
            if(P==0){zc++;}
        }
        if(zc==0) live++; hist[zc<39?zc:39]++;
    }
    printf("n=%2d  samples=%lld  nonzero-product fraction = %.4f%%   (speedup ceiling on product = %.2fx)\n",
           N,S,100.0*live/S, (double)S/live);
    printf("   #zero factors: "); for(int k=0;k<6;k++) printf("%d:%.1f%% ",k,100.0*hist[k]/S); printf("\n");
    return 0;
}
