/* =====================================================================
 * langford6.cu -- Godfrey en coordonnees de parite, sans code de Gray.
 *
 * Idee.  En separant les positions impaires (rangee o) et paires (rangee e),
 * un ecart PAIR i=2m vaut  A_{2m} = P_m(o) + Q_m(e)  : les deux rangees sont
 * SEPAREES.  Or seuls les ecarts pairs peuvent s'annuler (A_i = M-i mod 2),
 * donc la survie d'un point ne depend de o que par le vecteur constant P(o).
 *
 * Consequence : pour un e_hi fixe, la carte e_lo -> Q(e) ne depend pas de o
 * et se tabule une fois pour tout un bloc.  Mieux : on precalcule, pour chaque
 * ecart m et chaque valeur atteignable v, le BITMAP des e_lo tels que
 * Q_m(e_lo) = v.  Un thread (un o) obtient alors le masque des points morts
 * par un simple OU de 16 bitmaps -- ~1 instruction pour 32 points, contre
 * ~65 par point pour la mise a jour SWAR + test de la v5.
 *
 * Le groupe de Klein devient "negation independante de chaque rangee", donc
 * on epingle o_1 = e_1 = 0 et on multiplie par 4 (verifie par oe_ref.c).
 * ===================================================================== */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#define CHECK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d: %s\n",#x,__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(1);} }while(0)

#define ADD160(z,x) asm volatile( \
    "add.cc.u32  %0, %0, %5;\n\t addc.cc.u32 %1, %1, %6;\n\t"  \
    "addc.cc.u32 %2, %2, %7;\n\t addc.cc.u32 %3, %3, %8;\n\t"  \
    "addc.u32    %4, %4, %9;"                                   \
    : "+r"(z[0]),"+r"(z[1]),"+r"(z[2]),"+r"(z[3]),"+r"(z[4])    \
    : "r"(x[0]),"r"(x[1]),"r"(x[2]),"r"(x[3]),"r"(x[4]))
#define SUB160(z,x) asm volatile( \
    "sub.cc.u32  %0, %0, %5;\n\t subc.cc.u32 %1, %1, %6;\n\t"  \
    "subc.cc.u32 %2, %2, %7;\n\t subc.cc.u32 %3, %3, %8;\n\t"  \
    "subc.u32    %4, %4, %9;"                                   \
    : "+r"(z[0]),"+r"(z[1]),"+r"(z[2]),"+r"(z[3]),"+r"(z[4])    \
    : "r"(x[0]),"r"(x[1]),"r"(x[2]),"r"(x[3]),"r"(x[4]))

/* produit 160 bits des 32 voies SWAR, signe inclus dans la voie 31 */
__device__ __forceinline__ void prod160(const uint32_t *rr, uint32_t *z, int &neg)
{
    int p[16];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        uint32_t d = rr[k] ^ 0x80808080u;
        int v0=(int)(signed char)(d),      v1=(int)(signed char)(d>>8);
        int v2=(int)(signed char)(d>>16),  v3=(int)(signed char)(d>>24);
        p[2*k]=v0*v1; p[2*k+1]=v2*v3;
    }
    int q4[8];
    #pragma unroll
    for (int k=0;k<8;k++) q4[k]=p[2*k]*p[2*k+1];
    long long s[4];
    #pragma unroll
    for (int k=0;k<4;k++) s[k]=(long long)q4[2*k]*q4[2*k+1];
    neg = 0;
    unsigned long long u[4];
    #pragma unroll
    for (int k=0;k<4;k++){ neg ^= (s[k]<0); u[k]=(unsigned long long)(s[k]<0?-s[k]:s[k]); }
    uint32_t A[3], C[3];
    {   uint32_t a0=(uint32_t)u[0],a1=(uint32_t)(u[0]>>32);
        uint32_t c0=(uint32_t)u[1],c1=(uint32_t)(u[1]>>32);
        uint64_t p00=(uint64_t)a0*c0,p01=(uint64_t)a0*c1,p10=(uint64_t)a1*c0,p11=(uint64_t)a1*c1;
        uint64_t m=(p00>>32)+(uint32_t)p01+(uint32_t)p10;
        A[0]=(uint32_t)p00; A[1]=(uint32_t)m; A[2]=(uint32_t)((m>>32)+(p01>>32)+(p10>>32)+p11); }
    {   uint32_t a0=(uint32_t)u[2],a1=(uint32_t)(u[2]>>32);
        uint32_t c0=(uint32_t)u[3],c1=(uint32_t)(u[3]>>32);
        uint64_t p00=(uint64_t)a0*c0,p01=(uint64_t)a0*c1,p10=(uint64_t)a1*c0,p11=(uint64_t)a1*c1;
        uint64_t m=(p00>>32)+(uint32_t)p01+(uint32_t)p10;
        C[0]=(uint32_t)p00; C[1]=(uint32_t)m; C[2]=(uint32_t)((m>>32)+(p01>>32)+(p10>>32)+p11); }
    {   uint64_t P00=(uint64_t)A[0]*C[0];
        uint64_t P01=(uint64_t)A[0]*C[1], P10=(uint64_t)A[1]*C[0];
        uint64_t P02=(uint64_t)A[0]*C[2], P11=(uint64_t)A[1]*C[1], P20=(uint64_t)A[2]*C[0];
        uint64_t P12=(uint64_t)A[1]*C[2], P21=(uint64_t)A[2]*C[1];
        uint64_t P22=(uint64_t)A[2]*C[2];
        uint64_t cy,s0;
        s0=(uint32_t)P00;                                z[0]=(uint32_t)s0;
        cy=(s0>>32)+(P00>>32);
        s0=cy+(uint32_t)P01+(uint32_t)P10;               z[1]=(uint32_t)s0;
        cy=(s0>>32)+(P01>>32)+(P10>>32);
        s0=cy+(uint32_t)P02+(uint32_t)P11+(uint32_t)P20; z[2]=(uint32_t)s0;
        cy=(s0>>32)+(P02>>32)+(P11>>32)+(P20>>32);
        s0=cy+(uint32_t)P12+(uint32_t)P21;               z[3]=(uint32_t)s0;
        cy=(s0>>32)+(P12>>32)+(P21>>32);
        s0=cy+(uint32_t)P22;                             z[4]=(uint32_t)s0; }
}


/* NOTE (mesuree) : passer les 15 ecarts impairs a l'arbre sous forme
 * d'entiers, plutot que de les empaqueter en octets SWAR pour les re-extraire,
 * PARAIT economiser ~65 instructions par survivant.  En pratique c'est 1,85x
 * PLUS LENT (376 -> 203 Gsums/s) : garder 15 entiers vivants en meme temps que
 * l'arbre fait deborder le fichier de registres.  L'empaquetage SWAR n'est pas
 * du gaspillage, c'est de la compression de registres.                       */


/* Produit 160 bits a entree mixte : 16 ecarts PAIRS en SWAR (4 mots) et les 8
 * produits deja formes des ecarts IMPAIRS.  Les impairs sortent des popcounts
 * en entiers ; les empaqueter en octets pour les re-extraire aussitot coute
 * ~65 instructions par survivant.  On les multiplie donc DEUX A DEUX a la
 * volee : seuls 8 produits restent vivants (contre 15 entiers, ce qui faisait
 * deborder le fichier de registres -- essai mesure a 1,85x plus lent).       */
__device__ __forceinline__ void prod160m(const uint32_t *ev, const int *p8,
                                         uint32_t *z)
{
    int p[16];
    #pragma unroll
    for (int k=0;k<4;k++){
        uint32_t d = ev[k] ^ 0x80808080u;
        int v0=(int)(signed char)(d),     v1=(int)(signed char)(d>>8);
        int v2=(int)(signed char)(d>>16), v3=(int)(signed char)(d>>24);
        p[2*k]=v0*v1; p[2*k+1]=v2*v3;
    }
    #pragma unroll
    for (int k=0;k<8;k++) p[8+k]=p8[k];
    int q4[8];
    #pragma unroll
    for (int k=0;k<8;k++) q4[k]=p[2*k]*p[2*k+1];
    long long s[4];
    #pragma unroll
    for (int k=0;k<4;k++) s[k]=(long long)q4[2*k]*q4[2*k+1];

    /* Tout en complement a deux modulo 2^160 : l'accumulateur etant deja
     * modulaire et |somme finale| < 2^159, ni valeur absolue ni suivi de signe
     * ne sont necessaires (~24 instructions et une branche en moins).  Pour un
     * produit signe tronque il suffit de la correction
     *      a_s . b_s  =  a_u . b_u  -  2^w (s_a b_u + s_b a_u).              */
    uint32_t A[3], C[3];
    #pragma unroll
    for (int h=0;h<2;h++){
        const unsigned long long au=(unsigned long long)s[2*h];
        const unsigned long long cu=(unsigned long long)s[2*h+1];
        uint32_t a0=(uint32_t)au, a1=(uint32_t)(au>>32);
        uint32_t c0=(uint32_t)cu, c1=(uint32_t)(cu>>32);
        uint64_t p00=(uint64_t)a0*c0, p01=(uint64_t)a0*c1, p10=(uint64_t)a1*c0;
        uint32_t p11=a1*c1;
        uint64_t m=(p00>>32)+(uint32_t)p01+(uint32_t)p10;
        uint32_t r0=(uint32_t)p00, r1=(uint32_t)m;
        uint32_t r2=(uint32_t)((m>>32)+(p01>>32)+(p10>>32)+p11);
        r2 -= (s[2*h]  <0) ? c0 : 0u;
        r2 -= (s[2*h+1]<0) ? a0 : 0u;
        if(h==0){ A[0]=r0; A[1]=r1; A[2]=r2; } else { C[0]=r0; C[1]=r1; C[2]=r2; }
    }
    {   uint64_t P00=(uint64_t)A[0]*C[0];
        uint64_t P01=(uint64_t)A[0]*C[1], P10=(uint64_t)A[1]*C[0];
        uint64_t P02=(uint64_t)A[0]*C[2], P11=(uint64_t)A[1]*C[1], P20=(uint64_t)A[2]*C[0];
        uint64_t P12=(uint64_t)A[1]*C[2], P21=(uint64_t)A[2]*C[1];
        uint32_t P22=A[2]*C[2];
        uint64_t cy,s0;
        s0=(uint32_t)P00;                                z[0]=(uint32_t)s0;
        cy=(s0>>32)+(P00>>32);
        s0=cy+(uint32_t)P01+(uint32_t)P10;               z[1]=(uint32_t)s0;
        cy=(s0>>32)+(P01>>32)+(P10>>32);
        s0=cy+(uint32_t)P02+(uint32_t)P11+(uint32_t)P20; z[2]=(uint32_t)s0;
        cy=(s0>>32)+(P02>>32)+(P11>>32)+(P20>>32);
        s0=cy+(uint32_t)P12+(uint32_t)P21;               z[3]=(uint32_t)s0;
        cy=(s0>>32)+(P12>>32)+(P21>>32);
        z[4]=(uint32_t)(cy+P22);
        if ((int)A[2]<0){ uint64_t t=(uint64_t)z[3]-(uint64_t)C[0];
                          z[3]=(uint32_t)t; z[4]-=C[1]+(uint32_t)((t>>63)&1ull); }
        if ((int)C[2]<0){ uint64_t t=(uint64_t)z[3]-(uint64_t)A[0];
                          z[3]=(uint32_t)t; z[4]-=A[1]+(uint32_t)((t>>63)&1ull); }
    }
}

/* autocorrelation de rangee, lag m, sur N cellules */
template<int N> __device__ __forceinline__ int rowP(uint32_t r,int m){
    const int L=N-m; return L-2*__popc((r^(r>>m))&((1u<<L)-1u)); }


/* un survivant : (tid', e_lo') -> les 31 ecarts, puis le produit 160 bits */
#define LF6_ONE(ENT) do {                                                      \
    const uint32_t ent=(ENT); const int t2=(int)(ent>>16), el2=(int)(ent&0xFFFFu); \
    const uint32_t o2 = sPw[t2][4];                                             \
    const uint32_t e2 = ((vhi<<K) | (uint32_t)el2) << 1;                        \
    uint32_t ev[4]; int p8[8];                                                 \
    ev[0]=sT[el2][0]+sPw[t2][0]; ev[1]=sT[el2][1]+sPw[t2][1];                   \
    ev[2]=sT[el2][2]+sPw[t2][2]; ev[3]=sT[el2][3]+sPw[t2][3];                   \
    const int sgn = ((__popc(o2)+__popc(e2)) & 1) ? -1 : 1;                     \
    _Pragma("unroll")                                                          \
    for (int k=0;k<8;k++){                                                     \
        int xv0, xv1;                                                          \
        { const int m=2*k+1;                                                   \
          if (m > MO) xv0 = (m==MO+1) ? sgn : 1;                               \
          else { const int L1=N-m, L2=N-m-1;                                   \
            int s1=__popc((o2 ^ (e2>>m)) & ((1u<<L1)-1u));                      \
            int s2=(L2>0)?__popc((e2 ^ (o2>>(m+1))) & ((1u<<L2)-1u)):0;         \
            xv0 = L1 + (L2>0?L2:0) - 2*(s1+s2); } }                            \
        { const int m=2*k+2;                                                   \
          if (m > MO) xv1 = (m==MO+1) ? sgn : 1;                               \
          else { const int L1=N-m, L2=N-m-1;                                   \
            int s1=__popc((o2 ^ (e2>>m)) & ((1u<<L1)-1u));                      \
            int s2=(L2>0)?__popc((e2 ^ (o2>>(m+1))) & ((1u<<L2)-1u)):0;         \
            xv1 = L1 + (L2>0?L2:0) - 2*(s1+s2); } }                            \
        p8[k] = xv0*xv1;                                                       \
    }                                                                          \
    uint32_t z[5]; prod160m(ev,p8,z);                                          \
    ADD160(acc,z);                                                             \
  } while(0)
#define LF6_DRAIN()      do { LF6_ONE(sQ[warp][(head+lane)&63]);               \
                              head=(head+32)&63; qlen-=32; } while(0)
#define LF6_DRAIN_TAIL() do { if (lane<qlen) LF6_ONE(sQ[warp][(head+lane)&63]); \
                              head=(head+qlen)&63; qlen=0; } while(0)

template<int N, int K>
__global__ __launch_bounds__(256,5)
void oe_kernel(uint32_t vhi0, uint32_t vcnt, uint32_t blk0, uint32_t * __restrict__ out)
{
    const int ME = (N+1)/2;            /* ecarts pairs   m=1..ME */
    const int MO = N/2;                /* ecarts impairs m=1..MO */
    const int NL = 1<<K;               /* valeurs de e_lo        */
    const int W  = NL/32;              /* mots par tranche       */
    const int WP = W+1;                /* +1 : anti-conflit banc */
    const int NS = ME*(N+1);           /* tranches de bitmap     */

    __shared__ uint32_t sT[NL][5];   /* stride 5 : anti-conflit de bancs */     /* Q(e) empaquete SWAR, biais 64 */
    __shared__ uint32_t sB[NS*WP];     /* bitmaps par (lag, valeur)     */
    __shared__ uint32_t sPw[256][5]; /* idem */   /* P(o) de chaque thread         */
    __shared__ uint32_t sQ[8][64];     /* file des survivants, par warp */

    const int tid = threadIdx.x;
    const uint32_t u = (blockIdx.x+blk0)*256u + (uint32_t)tid;
    const uint32_t ALLB0 = (1u<<N)-1u;
    const uint32_t rvF = __brev(u<<1) >> (32-N);
    const uint32_t o   = (rvF & 1u) ? (rvF ^ ALLB0) : rvF;   /* o = f(F) */

    /* P(o) : constant pour tout le thread.  Biais 64 sur chaque octet ; la
     * table Q porte aussi 64, donc la somme porte exactement 128.        */
    uint32_t Pw[4] = {0x41414141u,0x41414141u,0x41414141u,0x41414141u};
    uint32_t soff[8];   /* 2 offsets de 16 bits par registre */
    #pragma unroll
    for (int m=1;m<=ME;m++){
        const int P = rowP<N>(o,m), l = m-1;
        Pw[l>>2] = (Pw[l>>2] & ~(0xFFu<<(8*(l&3)))) | ((uint32_t)(P+64)<<(8*(l&3)));
        { const uint32_t off = (uint32_t)(((m-1)*(N+1) + (((-P)+(N-m))>>1)) * WP);
          if (l&1) soff[l>>1] |= off<<16; else soff[l>>1] = off; }
    }

    sPw[tid][0]=Pw[0]; sPw[tid][1]=Pw[1]; sPw[tid][2]=Pw[2]; sPw[tid][3]=Pw[3];
    sPw[tid][4]=o;                              /* le drain lit o d'un autre thread */

    /* Reflexion.  En coordonnees de parite rho echange les deux rangees :
     * sigma(o,e) = (f(e), f(o)) avec f = canon o rev, canon = negation de la
     * rangee si son bit 0 vaut 1.  f est une involution, donc le representant
     * canonique est  e <= f(o)  -- et f(o) est CONSTANT pour le thread.     */
    const uint32_t Fo   = u<<1;                 /* F, dont o = f(F) */
    const uint32_t Fv   = u;
    const uint32_t Fvhi = Fv >> K, Fvlo = Fv & ((1u<<K)-1u);
    const int wlo = (int)(Fvlo>>5), blo = (int)(Fvlo&31);

    uint32_t acc[5] = {0,0,0,0,0};
    const int warp = tid>>5, lane = tid&31;
    const unsigned ltm = (1u<<lane)-1u;
    int head = 0, qlen = 0;

    for (uint32_t vv = 0; vv < vcnt; vv++) {
        const uint32_t vhi = vhi0 + vv;
        __syncthreads();
        /* --- phase 1 : table Q (biais 64) --- */
        for (int el = tid; el < NL; el += 256) {
            const uint32_t e = ((vhi<<K) | (uint32_t)el) << 1;
            uint32_t w[4] = {0x40404040u,0x40404040u,0x40404040u,0x40404040u};
            #pragma unroll
            for (int m=1;m<=ME;m++){
                const int Q = rowP<N>(e,m), l = m-1;
                w[l>>2] = (w[l>>2] & ~(0xFFu<<(8*(l&3)))) | ((uint32_t)(Q+64)<<(8*(l&3)));
            }
            sT[el][0]=w[0]; sT[el][1]=w[1]; sT[el][2]=w[2]; sT[el][3]=w[3];
        }
        for (int s = tid; s < NS*WP; s += 256) sB[s] = 0;
        __syncthreads();

        /* --- phase 2 : bitmaps.  Le thread possede le mot (m,w) en propre,
         * donc lecture-modification-ecriture simple, sans atomique.     */
        for (int task = tid; task < ME*W; task += 256) {
            const int m = task / W + 1, w = task % W;
            const int base = (m-1)*(N+1);
            for (int b=0;b<32;b++){
                const int el = w*32+b;
                const int Q = (int)((sT[el][(m-1)>>2] >> (8*((m-1)&3))) & 0xFFu) - 64;
                sB[(base + ((Q + (N-m))>>1))*WP + w] |= 1u<<b;
            }
        }
        __syncthreads();

        /* --- phase 3 : balayage.  Un OU de ME bitmaps donne la mortalite
         * de 32 points d'un coup ; les survivants sont compactes par warp
         * puis draines par paquets de 32 (sinon le warp tourne au MAX des
         * survivants et non a leur moyenne : c'est ce qui coutait 2x).   */
        for (int w=0; w<W; w++){
            uint32_t dead = sB[(soff[0]&0xFFFFu)+w];
            #pragma unroll
            for (int l=1;l<ME;l++)
                dead |= sB[(((l&1)?(soff[l>>1]>>16):(soff[l>>1]&0xFFFFu)))+w];
            uint32_t live = ~dead;
            if (vhi > Fvhi) live = 0u;
            else if (vhi == Fvhi) {
                if (w > wlo) live = 0u;
                else if (w == wlo) live &= blo ? ((1u<<blo)-1u) : 0u;
            }
            while (__any_sync(0xffffffffu, live != 0u)) {
                const bool has = (live != 0u);
                int el = 0;
                if (has) { const int b = __ffs(live)-1; live &= live-1; el = w*32+b; }
                const unsigned bal = __ballot_sync(0xffffffffu, has);
                if (has) sQ[warp][(head+qlen+__popc(bal&ltm))&63]
                             = ((uint32_t)tid<<16) | (uint32_t)el;
                qlen += __popc(bal);
                if (qlen >= 32) { __syncwarp(); LF6_DRAIN(); }
            }
        }
        if (qlen) { __syncwarp(); LF6_DRAIN_TAIL(); }   /* vider avant le vhi suivant */
    }

    /* poids 2 sur toute la moitie canonique, poids 1 sur la diagonale */
    acc[4]=(acc[4]<<1)|(acc[3]>>31); acc[3]=(acc[3]<<1)|(acc[2]>>31);
    acc[2]=(acc[2]<<1)|(acc[1]>>31); acc[1]=(acc[1]<<1)|(acc[0]>>31);
    acc[0]<<=1;

    /* --- reduction de bloc (on reutilise sB) --- */
    __syncthreads();
    uint32_t (*sm)[5] = (uint32_t(*)[5])sB;
    #pragma unroll
    for (int k=0;k<5;k++) sm[tid][k]=acc[k];
    __syncthreads();
    for (int st=128; st; st>>=1){
        if (tid<st){ uint32_t x[5],y[5];
            #pragma unroll
            for(int k=0;k<5;k++){x[k]=sm[tid][k];y[k]=sm[tid+st][k];}
            ADD160(x,y);
            #pragma unroll
            for(int k=0;k<5;k++) sm[tid][k]=x[k]; }
        __syncthreads();
    }
    if (tid==0)
        #pragma unroll
        for (int k=0;k<5;k++) out[blockIdx.x*5+k]=sm[0][k];
}


/* Les orbites fixes de sigma : e = f(o), un point par o, poids 1.  Sorties du
 * noyau principal pour liberer 5 registres (l'occupancy y est le facteur
 * limitant) ; il y en a 2^(N-1) en tout, soit 1e-9 du travail.              */
template<int N>
__global__ __launch_bounds__(256)
void diag_kernel(uint32_t * __restrict__ out)
{
    const int ME=(N+1)/2, MO=N/2;
    const uint32_t u = blockIdx.x*256u + threadIdx.x;
    const uint32_t ALLB=(1u<<N)-1u;
    const uint32_t rvF=__brev(u<<1)>>(32-N);
    const uint32_t o = (rvF&1u)?(rvF^ALLB):rvF;
    const uint32_t e = u<<1;
    uint32_t r[8]; r[0]=r[1]=r[2]=r[3]=r[4]=r[5]=r[6]=r[7]=0x81818181u;
    #pragma unroll
    for (int m=1;m<=ME;m++){
        const int A = rowP<N>(o,m)+rowP<N>(e,m), l=m-1;
        r[l>>2] = (r[l>>2] & ~(0xFFu<<(8*(l&3)))) | ((uint32_t)(A+128)<<(8*(l&3)));
    }
    #pragma unroll
    for (int m=1;m<=MO;m++){
        const int L1=N-m, L2=N-m-1;
        int X = L1 - 2*__popc((o ^ (e>>m)) & ((1u<<L1)-1u));
        if (L2>0) X += L2 - 2*__popc((e ^ (o>>(m+1))) & ((1u<<L2)-1u));
        const int nl=16+(m-1);
        r[nl>>2] = (r[nl>>2] & ~(0xFFu<<(8*(nl&3)))) | ((uint32_t)(X+128)<<(8*(nl&3)));
    }
    if ((__popc(o)+__popc(e))&1) r[7] ^= 0xFE000000u;
    uint32_t acc[5]={0,0,0,0,0}, z[5]; int neg;
    prod160(r,z,neg);
    if (neg) SUB160(acc,z); else ADD160(acc,z);
    __shared__ uint32_t sm[256][5];
    #pragma unroll
    for (int k=0;k<5;k++) sm[threadIdx.x][k]=acc[k];
    __syncthreads();
    for (int st=128; st; st>>=1){
        if (threadIdx.x<st){ uint32_t x[5],y[5];
            #pragma unroll
            for(int k=0;k<5;k++){x[k]=sm[threadIdx.x][k];y[k]=sm[threadIdx.x+st][k];}
            ADD160(x,y);
            #pragma unroll
            for(int k=0;k<5;k++) sm[threadIdx.x][k]=x[k]; }
        __syncthreads();
    }
    if (threadIdx.x==0)
        #pragma unroll
        for (int k=0;k<5;k++) out[blockIdx.x*5+k]=sm[0][k];
}

/* ====================== hote ====================== */
typedef struct { uint32_t w[5]; } u160;
static void h_add(u160*z,const u160*x){ uint64_t c=0;
    for(int k=0;k<5;k++){ uint64_t s=(uint64_t)z->w[k]+x->w[k]+c; z->w[k]=(uint32_t)s; c=s>>32; } }
static void h_shl2(u160*z){ uint32_t c=0;
    for(int k=0;k<5;k++){ uint32_t v=z->w[k]; z->w[k]=(v<<2)|c; c=v>>30; } }
static void print_u160(u160 v){ char o[64]; int len=0; uint32_t t[5]; memcpy(t,v.w,sizeof t);
    for(;;){ uint64_t rem=0;
        for(int k=4;k>=0;k--){ uint64_t c=(rem<<32)|t[k]; t[k]=(uint32_t)(c/1000000000ULL); rem=c%1000000000ULL; }
        int nz=0; for(int k=0;k<5;k++) if(t[k]) nz=1;
        for(int d=0;d<9;d++){ o[len++]='0'+(int)(rem%10); rem/=10; }
        if(!nz) break; }
    while(len>1&&o[len-1]=='0') len--;
    for(int k=len-1;k>=0;k--) putchar(o[k]); }
static void print_i160(u160 v){ if(v.w[4]&0x80000000u){ putchar('-'); uint64_t c=1;
        for(int k=0;k<5;k++){ uint64_t s=(uint64_t)(~v.w[k])+c; v.w[k]=(uint32_t)s; c=s>>32; } }
    print_u160(v); }

typedef void (*kern_t)(uint32_t,uint32_t,uint32_t,uint32_t*);
typedef void (*dker_t)(uint32_t*);
#define DINST(NN) case NN: return (dker_t)diag_kernel<NN>;
static dker_t dpick(int N){ switch(N){
    DINST(11) DINST(12) DINST(15) DINST(16) DINST(19) DINST(20) DINST(23) DINST(24)
    DINST(27) DINST(28) DINST(31) default: return NULL; } }
#define K_ 7
#define INST(NN) case NN: return (kern_t)oe_kernel<NN,K_>;
static kern_t pick(int N){ switch(N){
    INST(11) INST(12) INST(15) INST(16) INST(19) INST(20) INST(23) INST(24)
    INST(27) INST(28) INST(31) default: return NULL; } }

int main(int argc,char**argv){
    int N=15; long long from=0,count=-1; int chunk=16;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-n")) N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--from")) from=atoll(argv[++i]);
        else if(!strcmp(argv[i],"--count")) count=atoll(argv[++i]);
        else if(!strcmp(argv[i],"--chunk")) chunk=atoi(argv[++i]);
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; } }

    const int FREE = N-1;                       /* bits libres par rangee */
    if (FREE < K_ || FREE < 8){ fprintf(stderr,"n trop petit pour ce decoupage\n"); return 1; }
    const long long nvhi  = 1LL<<(FREE-K_);
    const int blocks = 1<<(FREE-8);
    if(count<0) count=nvhi-from;

    kern_t Kf=pick(N); if(!Kf){ fprintf(stderr,"N non instancie\n"); return 1; }
    uint32_t *d_out,*h_out;
    CHECK(cudaMalloc(&d_out,(size_t)blocks*5*4));
    h_out=(uint32_t*)malloc((size_t)blocks*5*4);

    fprintf(stderr,"n=%d  rangees de %d cases  K=%d  blocs=%d  vhi=%lld  points=2^%d\n",
            N,N,K_,blocks,nvhi,2*FREE);

    u160 total; memset(&total,0,sizeof total);
    struct timespec T0,T1; clock_gettime(CLOCK_MONOTONIC,&T0);
    long long done=0;
    for(long long s=from;s<from+count;s+=chunk){
        int c=(int)((s+chunk<=from+count)?chunk:(from+count-s));
        /* seuls les blocs avec u >= v peuvent contribuer (predicat v <= u) */
        long long b0 = ((long long)s<<K_)>>8; if(b0<0) b0=0;
        int nb = blocks-(int)b0; if(nb<=0){ done+=c; continue; }
        Kf<<<nb,256>>>((uint32_t)s,(uint32_t)c,(uint32_t)b0,d_out);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_out,d_out,(size_t)blocks*5*4,cudaMemcpyDeviceToHost));
        for(int q=0;q<nb;q++) h_add(&total,(u160*)(h_out+5*q));
        done+=c;
    }
    if(from==0&&done==nvhi){          /* orbites fixes : un seul lancement */
        dker_t Dk=dpick(N);
        Dk<<<blocks,256>>>(d_out);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_out,d_out,(size_t)blocks*5*4,cudaMemcpyDeviceToHost));
        for(int q=0;q<blocks;q++) h_add(&total,(u160*)(h_out+5*q));
    }
    clock_gettime(CLOCK_MONOTONIC,&T1);
    double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)*1e-9;
    double pts=(double)done*(double)(1<<K_)*(double)(1<<FREE);
    fprintf(stderr,"%lld/%lld vhi, %.6g points en %.1f s -> %.4f Gsums/s\n",
            done,nvhi,pts,el,pts/el*1e-9);

    if(from==0&&done==nvhi){
        h_shl2(&total);                          /* x4 : groupe de Klein */
        int ok=1; for(int b=0;b<2*N;b++) if((total.w[b>>5]>>(b&31))&1) ok=0;
        if(!ok){ fprintf(stderr,"*** AUTO-TEST ECHOUE : non divisible par 2^%d ***\n",2*N); return 1; }
        u160 v; memset(&v,0,sizeof v);
        for(int b=2*N;b<160;b++) if((total.w[b>>5]>>(b&31))&1) v.w[(b-2*N)>>5]|=1u<<((b-2*N)&31);
        printf("n=%d   V(n) = 2*L(2,n) = ",N); print_u160(v); printf("\n");
        uint32_t c=0; u160 h;
        for(int q=4;q>=0;q--){ uint32_t x=v.w[q]; h.w[q]=(x>>1)|(c<<31); c=x&1; }
        printf("n=%d   L(2,%d)         = ",N,N); print_u160(h); printf("\n");
    } else { printf("PARTIAL n=%d vhi %lld..%lld sum = ",N,from,from+done-1);
             print_i160(total); printf("\n"); }
    return 0;
}
