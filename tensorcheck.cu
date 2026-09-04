/* Debit brut de cette carte, mesure et non estime :
 *   (a) tensor cores INT8, mma.m16n8k32  -> MAC/s
 *   (b) le motif exact du drain : xor + masque + popc  -> MAC binaires/s
 * Un popcount sur un XOR 32 bits EST un produit scalaire binaire de longueur
 * 32 : c'est le vrai concurrent du tensor core sur des donnees +-1.        */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define ITER 4096

__global__ void k_mma(int *out){
    uint32_t a[4]={0x01020304u,0x05060708u,0x090a0b0cu,0x0d0e0f10u};
    uint32_t b[2]={0x11121314u,0x15161718u};
    int c[4]={0,0,0,0};
    a[0]^=threadIdx.x; b[0]^=threadIdx.x;
    #pragma unroll 8
    for(int i=0;i<ITER;i++){
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                     :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
    if(out) out[threadIdx.x]=c[0]+c[1]+c[2]+c[3];
}

/* 8 correlations independantes par tour, comme dans le drain */
__global__ void k_popc(int *out){
    uint32_t o=0x5A5A5A5Au^threadIdx.x, e=0xC3C3C3C3u^(threadIdx.x<<3);
    int s=0;
    #pragma unroll 8
    for(int i=0;i<ITER;i++){
        #pragma unroll
        for(int m=1;m<=8;m++) s+=__popc((o ^ (e>>m)) & ((1u<<(31-m))-1u));
        o+=s; e^=s;
    }
    if(out) out[threadIdx.x]=s;
}

static float run(void(*kern)(int*),int blocks){
    cudaEvent_t A,B; cudaEventCreate(&A); cudaEventCreate(&B);
    kern<<<blocks,256>>>(nullptr); cudaDeviceSynchronize();
    cudaEventRecord(A); for(int r=0;r<10;r++) kern<<<blocks,256>>>(nullptr);
    cudaEventRecord(B); cudaEventSynchronize(B);
    float ms; cudaEventElapsedTime(&ms,A,B); return ms/10.0f;
}

int main(){
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    int blocks=p.multiProcessorCount*8;
    printf("GPU : %s, %d SM\n",p.name,p.multiProcessorCount);

    float t=run(k_mma,blocks);
    /* chaque mma m16n8k32 = 16*8*32 MAC, par WARP */
    double warps=(double)blocks*256/32;
    double mac=warps*ITER*16.0*8.0*32.0;
    printf("tensor INT8 : %.3f ms  ->  %.2f e12 MAC/s\n", t, mac/(t*1e-3)/1e12);

    t=run(k_popc,blocks);
    /* chaque popc traite 31-m ~ 27 positions en moyenne ; 8 par tour */
    double lanes=(double)blocks*256;
    double bmac=lanes*ITER*8.0*27.0;
    printf("popc (xor+masque+popc, le motif du drain) : %.3f ms  ->  %.2f e12 MAC binaires/s\n",
           t, bmac/(t*1e-3)/1e12);
    return 0;
}
