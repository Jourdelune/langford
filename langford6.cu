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
 *
 * ---------------------------------------------------------------------
 * v7 -- les ecarts IMPAIRS deviennent eux aussi une somme de deux tables.
 *
 * Les 15 ecarts impairs sont BILINEAIRES en (o,e) : ils ne se separent pas
 * comme les pairs, et jusqu'ici le drain les recalculait par 22 popcounts et
 * par survivant -- 117 des 255 instructions.  Ils se separent quand meme, mais
 * sur un axe different, et c'est __brev qui l'offre :
 *
 *   o vient de  rvF = __brev(u<<1) >> (32-N)  avec  u = blockIdx*256 + tid,
 *   donc les bits FAIBLES de u -- exactement threadIdx -- atterrissent dans
 *   les bits FORTS de o.  A l'interieur d'un bloc, o ne varie que sur les
 *   HUIT bits [N-9, N-2] ; tout le reste de o est constant sur le bloc.
 *
 * Un ecart impair est une somme de produits O_a.E_c.  On classe les couples
 * (a,c) selon que a tombe dans la fenetre du thread et c dans e_lo :
 *
 *   a hors fenetre, c hors e_lo -> (bloc, e_hi)   \ ensemble : A(o, e_lo=0),
 *   a dans fenetre, c hors e_lo -> (thread, e_hi) /  un mot SWAR de sPw,
 *                                                    calcule une fois par vhi
 *   a hors fenetre, c dans e_lo -> (bloc, e_lo)   -> table sF, construite une
 *                                                    fois PAR BLOC (elle ne
 *                                                    depend pas de e_hi)
 *   a dans fenetre, c dans e_lo -> trois termes seulement a n=31/K=7, traites
 *                                  au vol par oe_fix
 *
 * Le drain lit donc  sF[e_lo] + sPw[thread]  en octets SWAR, exactement comme
 * pour les ecarts pairs, et ne fait plus AUCUN popcount.  Les 30 popcounts qui
 * etaient par survivant sont devenus par (thread, vhi), donc amortis sur les
 * ~16,5 survivants du thread.  Le 16e octet porte le signe : sPw y met
 * popc(o)+popc(e_hi), sF y met popc(e_lo), et la parite de la somme decide.
 *
 * Corps de la boucle interne : 288 -> 225 instructions SASS, dont 26 POPC -> 0.
 * Decomposition verifiee exactement par check_decomp.c (0 divergence sur
 * 3.10^6 ecarts a n=31, et pour n = 11..28).
 *
 * S'y ajoutent deux choses qui ne changent pas le resultat au bit pres :
 *   - un chemin rapide dans l'arbre de produit quand les quatre facteurs de
 *     64 bits tiennent sur un int32 (le terme typique vaut 2^64, le pire cas
 *     2^169) : mul64_160s remplace deux mul64_96 plus mul96_160 ;
 *   - des strides de 6 et 10 mots, multiples de 8 octets et de moitie impaire,
 *     qui rendent les huit mots SWAR lisibles en quatre LDS.64 sans conflit.
 *
 * Mesure appariee contre la v6.1, GPU au repos, graine de tirage fixe :
 * 657,3 / 663,1 h  ->  551,6 / 551,5 h a n=31, soit +19,7 %.  Les sommes
 * partielles sont IDENTIQUES AU BIT PRES : les lignes PART= deja produites
 * par la v6/v6.1 restent valides et s'agregent sans reserve.
 * ===================================================================== */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cmath>
#ifndef LF6_MINBLK
#define LF6_MINBLK 3          /* mesure : 3 supprime le spill sur sm_120 (+6,5 %)
                              *          et reste neutre sur sm_89           */
#endif
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

/* Extraction SIGNEE d'un octet SWAR en UNE instruction.
 *
 * `prmt.b32` recopie quatre octets choisis parmi les huit de (a,b) ; quand le
 * bit 3 d'un selecteur est mis, l'octet de sortie vaut 0x00 ou 0xFF selon le
 * SIGNE de l'octet choisi.  Un selecteur (8|j,8|j,8|j,j) etend donc l'octet j
 * en un entier 32 bits signe, seul.
 *
 * nvcc ne connait ce motif que pour l'octet 0 ; pour les octets 1 et 2 il
 * emet decalage + xor + decalage arithmetique, soit 3 instructions au lieu
 * d'une.  Mesure : 9 instructions par mot SWAR au lieu de 5, sur 4 mots et
 * pour chaque survivant.                                                    */
#define SEXTB0 0x8880
#define SEXTB1 0x9991
#define SEXTB2 0xAAA2
#define SEXTB3 0xBBB3
template<int SEL> __device__ __forceinline__ int sextb(uint32_t d){
    int r; asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(d), "n"(SEL)); return r; }

/* produit 160 bits des 32 voies SWAR, signe inclus dans la voie 31 */
__device__ __forceinline__ void prod160(const uint32_t *rr, uint32_t *z, int &neg)
{
    int p[16];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        uint32_t d = rr[k] ^ 0x80808080u;
        int v0=sextb<SEXTB0>(d), v1=sextb<SEXTB1>(d);
        int v2=sextb<SEXTB2>(d), v3=sextb<SEXTB3>(d);
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


/* ---------------------------------------------------------------------------
 * Multiplication multi-mots par CHAINES DE RETENUE (mad.lo.cc / madc.hi.cc).
 *
 * La formulation naturelle en `uint64_t` -- former les produits partiels
 * 32x32->64 puis recombiner les moities a la main -- oblige ptxas a
 * materialiser chaque produit dans une paire de registres, puis a propager les
 * retenues par des IADD3/IADD3.X separes : ~21 instructions de propagation
 * pour l'etage final.  Les formes `mad{c}.{lo,hi}.cc` du PTX ABSORBENT
 * l'addition ET la retenue dans le multiplieur : une IMAD par produit partiel,
 * zero addition explicite.
 *
 * Contrainte : le drapeau de retenue n'est pas modelise par le compilateur.
 * Chaque chaine doit donc tenir dans UN SEUL bloc asm, et les sorties sont en
 * "=&r" (earlyclobber) -- sans quoi le repartiteur peut aliaser une sortie
 * ecrite tot avec une entree relue plus tard dans la chaine.
 * ------------------------------------------------------------------------- */

/* 64x64 -> 96 bits (tronque), non signe. 7 instructions. */
__device__ __forceinline__ void mul64_96(uint32_t a0,uint32_t a1,
                                         uint32_t c0,uint32_t c1,uint32_t*r)
{
    asm("{\n\t"
        "mul.lo.u32     %0, %3, %5;\n\t"
        "mul.hi.u32     %1, %3, %5;\n\t"
        "mad.lo.cc.u32  %1, %4, %5, %1;\n\t"
        "madc.hi.u32    %2, %4, %5, 0;\n\t"
        "mad.lo.cc.u32  %1, %3, %6, %1;\n\t"
        "madc.hi.u32    %2, %3, %6, %2;\n\t"
        "mad.lo.u32     %2, %4, %6, %2;\n\t"
        "}" : "=&r"(r[0]),"=&r"(r[1]),"=&r"(r[2])
            : "r"(a0),"r"(a1),"r"(c0),"r"(c1));
}

/* 96x96 -> 160 bits (tronque), non signe.  Schoolbook ligne par ligne : pour
 * chaque limbe c_j, une passe sur les moities BASSES puis une passe sur les
 * HAUTES, chacune sa propre chaine de retenue (les deux ne peuvent pas
 * s'entrelacer, il n'y a qu'un drapeau).  Les retenues sortant du limbe 4 sont
 * jetees : c'est la troncature, elle est gratuite.  17 instructions contre
 * ~30 pour la version uint64_t. */
__device__ __forceinline__ void mul96_160(const uint32_t*A,const uint32_t*C,uint32_t*z)
{
    asm("{\n\t"
        /* ligne c0 -- accumulateur nul, une seule chaine suffit */
        "mul.lo.u32     %0, %5,  %8;\n\t"
        "mul.hi.u32     %1, %5,  %8;\n\t"
        "mad.lo.cc.u32  %1, %6,  %8, %1;\n\t"
        "madc.hi.u32    %2, %6,  %8, 0;\n\t"
        "mad.lo.cc.u32  %2, %7,  %8, %2;\n\t"
        "madc.hi.u32    %3, %7,  %8, 0;\n\t"
        /* ligne c1 : moities basses dans z1..z3, la retenue INITIALISE z4 */
        "mad.lo.cc.u32  %1, %5,  %9, %1;\n\t"
        "madc.lo.cc.u32 %2, %6,  %9, %2;\n\t"
        "madc.lo.cc.u32 %3, %7,  %9, %3;\n\t"
        "addc.u32       %4, 0, 0;\n\t"
        /* ligne c1 : moities hautes dans z2..z4 */
        "mad.hi.cc.u32  %2, %5,  %9, %2;\n\t"
        "madc.hi.cc.u32 %3, %6,  %9, %3;\n\t"
        "madc.hi.u32    %4, %7,  %9, %4;\n\t"
        /* ligne c2 : basses dans z2..z4, hautes dans z3..z4 (hi(a2c2) -> z5) */
        "mad.lo.cc.u32  %2, %5, %10, %2;\n\t"
        "madc.lo.cc.u32 %3, %6, %10, %3;\n\t"
        "madc.lo.u32    %4, %7, %10, %4;\n\t"
        "mad.hi.cc.u32  %3, %5, %10, %3;\n\t"
        "madc.hi.u32    %4, %6, %10, %4;\n\t"
        "}" : "=&r"(z[0]),"=&r"(z[1]),"=&r"(z[2]),"=&r"(z[3]),"=&r"(z[4])
            : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(C[0]),"r"(C[1]),"r"(C[2]));
}

/* 64x64 -> 128 bits SIGNE, etendu sur 160.  Les deux operandes tiennent sur un
 * int64, donc le produit tient sur 126 bits : rien n'est tronque, l'extension
 * de signe suffit.  Une seule chaine de retenue, un seul bloc asm.          */
__device__ __forceinline__ void mul64_160s(long long A, long long C, uint32_t *z)
{
    const uint32_t a0=(uint32_t)(unsigned long long)A, a1=(uint32_t)((unsigned long long)A>>32);
    const uint32_t c0=(uint32_t)(unsigned long long)C, c1=(uint32_t)((unsigned long long)C>>32);
    asm("{\n\t"
        ".reg .u32 t0, t1, ma, mc;\n\t"
        "mul.lo.u32     %0, %5, %7;\n\t"
        "mul.hi.u32     %1, %5, %7;\n\t"
        "mad.lo.cc.u32  %1, %5, %8, %1;\n\t"
        "madc.hi.u32    %2, %5, %8, 0;\n\t"
        "mad.lo.cc.u32  %1, %6, %7, %1;\n\t"
        "madc.hi.cc.u32 %2, %6, %7, %2;\n\t"
        "addc.u32       %3, 0, 0;\n\t"
        "mad.lo.cc.u32  %2, %6, %8, %2;\n\t"
        "madc.hi.u32    %3, %6, %8, %3;\n\t"
        /* a_s.c_s = a_u.c_u - 2^64 (s_a.c_u + s_c.a_u) */
        "shr.s32        ma, %6, 31;\n\t"
        "shr.s32        mc, %8, 31;\n\t"
        "and.b32        t0, %7, ma;\n\t"
        "and.b32        t1, %8, ma;\n\t"
        "sub.cc.u32     %2, %2, t0;\n\t"
        "subc.u32       %3, %3, t1;\n\t"
        "and.b32        t0, %5, mc;\n\t"
        "and.b32        t1, %6, mc;\n\t"
        "sub.cc.u32     %2, %2, t0;\n\t"
        "subc.u32       %3, %3, t1;\n\t"
        "shr.s32        %4, %3, 31;\n\t"
        "}" : "=&r"(z[0]),"=&r"(z[1]),"=&r"(z[2]),"=&r"(z[3]),"=&r"(z[4])
            : "r"(a0),"r"(a1),"r"(c0),"r"(c1));
}

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
        int v0=sextb<SEXTB0>(d), v1=sextb<SEXTB1>(d);
        int v2=sextb<SEXTB2>(d), v3=sextb<SEXTB3>(d);
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

    /* Chemin rapide.  Le pire cas exige 160 bits (un terme peut valoir 2^169),
     * mais le terme TYPIQUE vaut ~2^64 : |A_i| ~ sqrt(2n-i), donc l'esperance
     * de log2 du produit est 64,5 avec un ecart-type de 8,9.  Quand les quatre
     * s[k] tiennent sur un int32 -- test exact, deux instructions chacun --
     * A = s0.s1 et C = s2.s3 tiennent sur un int64 et |A.C| < 2^124 : une
     * multiplication 64x64 remplace mul64_96 deux fois PLUS mul96_160.  Le
     * chemin lent reste la pour le reste, donc le resultat est inchange au bit
     * pres ; seul le nombre d'instructions bouge.                            */
    {   uint32_t bad = 0;
        #pragma unroll
        for (int k=0;k<4;k++){
            const unsigned long long u=(unsigned long long)s[k];
            bad |= (uint32_t)(u>>32) ^ (uint32_t)((int32_t)(uint32_t)u>>31);
        }
        if (!bad) {
            mul64_160s((long long)(int)s[0]*(int)s[1],
                       (long long)(int)s[2]*(int)s[3], z);
            return;
        }
    }

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
        uint32_t r[3]; mul64_96(a0,a1,c0,c1,r);
        r[2] -= (s[2*h]  <0) ? c0 : 0u;
        r[2] -= (s[2*h+1]<0) ? a0 : 0u;
        if(h==0){ A[0]=r[0]; A[1]=r[1]; A[2]=r[2]; } else { C[0]=r[0]; C[1]=r[1]; C[2]=r[2]; }
    }
    mul96_160(A,C,z);
    /* correction de signe du dernier etage : -2^96 (s_A.C + s_C.A), tronquee
     * a 160 bits, donc seulement les limbes 3 et 4.                        */
    if ((int)A[2]<0){ uint64_t t=(uint64_t)z[3]-(uint64_t)C[0];
                      z[3]=(uint32_t)t; z[4]-=C[1]+(uint32_t)((t>>63)&1ull); }
    if ((int)C[2]<0){ uint64_t t=(uint64_t)z[3]-(uint64_t)A[0];
                      z[3]=(uint32_t)t; z[4]-=A[1]+(uint32_t)((t>>63)&1ull); }
}

/* Correction (o_var x e_lo) et slot de signe.
 *
 * Deroule a la compilation.  A n=31/K=7 la boucle en c ne retient que TROIS
 * termes (m=14 c=7, m=15 c=6, m=15 c=7) ; ailleurs elle rend v tel quel et ne
 * coute rien.  Pour L > MO on rend 1 et le sextb correspondant est elimine.  */
template<int N,int K,int L>
__device__ __forceinline__ int oe_fix(int v, uint32_t o2, uint32_t e2)
{
    const int MO=N/2;
    if (L >  MO) return 1;                 /* bourrage */
    if (L == MO) return 1-2*(v&1);         /* signe : v = popc(o)+popc(e) */
    const int m=L+1, A0=N-9, A1=N-2;       /* [A0,A1] : les bits de o portes par tid */
    int t=0;
    #pragma unroll
    for (int c=1;c<=K;c++){
        if (c>=m && c-m>=A0 && c-m<=A1)                  /* partenaire de la 1re somme */
            t += (int)(((o2>>(c-m)) & (e2>>c)) & 1u);
        if (c<=N-m-2 && c+m+1>=A0 && c+m+1<=A1)          /* partenaire de la 2e somme */
            t += (int)(((o2>>(c+m+1)) & (e2>>c)) & 1u);
    }
    return t ? v+4*t : v;
}

/* autocorrelation de rangee, lag m, sur N cellules */
template<int N> __device__ __forceinline__ int rowP(uint32_t r,int m){
    const int L=N-m; return L-2*__popc((r^(r>>m))&((1u<<L)-1u)); }


/* lecture partagee 64 bits : deux mots SWAR contigus en une LDS.64 */
__device__ __forceinline__ uint2 LF6_LD2(const uint32_t *p){
    return *reinterpret_cast<const uint2*>(p); }

/* un mot SWAR d'ecarts impairs : quatre octets signes, deux produits. */
#define LF6_ODD(k) do {                                                        \
    const uint32_t d_ = (od_[k]) ^ 0x80808080u;                                \
    p8[2*(k)  ] = oe_fix<N,K,4*(k)+0>(sextb<SEXTB0>(d_),o2,e2)                 \
                * oe_fix<N,K,4*(k)+1>(sextb<SEXTB1>(d_),o2,e2);                \
    p8[2*(k)+1] = oe_fix<N,K,4*(k)+2>(sextb<SEXTB2>(d_),o2,e2)                 \
                * oe_fix<N,K,4*(k)+3>(sextb<SEXTB3>(d_),o2,e2);                \
  } while(0)

/* un survivant : (tid', e_lo') -> les 31 ecarts, puis le produit 160 bits.
 *
 * v7 : les ecarts IMPAIRS sortent d'une SOMME DE DEUX TABLES exactement comme
 * les pairs -- plus un seul popcount ici.                                    */
#define LF6_ONE(ENT) do {                                                      \
    const uint32_t ent=(ENT); const int t2=(int)(ent>>16), el2=(int)(ent&0xFFFFu); \
    const uint32_t e2 = ((vhi<<K) | (uint32_t)el2) << 1;                        \
    const uint32_t o2 = sPw[t2][4];                                             \
    uint32_t ev[4]; int p8[8];                                                 \
    { const uint2 ta=LF6_LD2(&sT[el2][0]),  tb=LF6_LD2(&sT[el2][2]);            \
      const uint2 pa=LF6_LD2(&sPw[t2][0]),  pb=LF6_LD2(&sPw[t2][2]);            \
      ev[0]=ta.x+pa.x; ev[1]=ta.y+pa.y; ev[2]=tb.x+pb.x; ev[3]=tb.y+pb.y; }     \
    uint32_t od_[4];                                                           \
    { const uint2 fa=LF6_LD2(&sF[el2][0]),  fb=LF6_LD2(&sF[el2][2]);            \
      const uint2 ga=LF6_LD2(&sPw[t2][6]),  gb=LF6_LD2(&sPw[t2][8]);            \
      od_[0]=fa.x+ga.x; od_[1]=fa.y+ga.y; od_[2]=fb.x+gb.x; od_[3]=fb.y+gb.y; } \
    LF6_ODD(0); LF6_ODD(1); LF6_ODD(2); LF6_ODD(3);                            \
    uint32_t z[5]; prod160m(ev,p8,z);                                          \
    ADD160(acc,z);                                                             \
  } while(0)
#define LF6_DRAIN()      do { LF6_ONE(sQ[warp][(head+lane)&63]);               \
                              head=(head+32)&63; qlen-=32; } while(0)
#define LF6_DRAIN_TAIL() do { if (lane<qlen) LF6_ONE(sQ[warp][(head+lane)&63]); \
                              head=(head+qlen)&63; qlen=0; } while(0)

template<int N, int K>
__global__ __launch_bounds__(256,LF6_MINBLK)
void oe_kernel(uint32_t vhi0, uint32_t vcnt, uint32_t blk0, uint32_t * __restrict__ out)
{
    const int ME = (N+1)/2;            /* ecarts pairs   m=1..ME */
    const int MO = N/2;                /* ecarts impairs m=1..MO */
    const int NL = 1<<K;               /* valeurs de e_lo        */
    const int W  = NL/32;              /* mots par tranche       */
    const int WP = W+1;                /* +1 : anti-conflit banc */
    const int NS = ME*(N+1);           /* tranches de bitmap     */
    /* Les 16 emplacements SWAR des ecarts IMPAIRS : MO ecarts (octets 0..MO-1),
     * le slot de signe (octet MO), le reste en bourrage.  MO <= 15 pour tout
     * n <= 31, donc quatre mots suffisent toujours. */
    const int OW = 4;

    /* Strides de 6 et 10 mots : multiples de 8 octets (donc LDS.64 legal) et
     * moitie IMPAIRE (3 et 5), donc les paires de bancs restent distinctes sur
     * seize voies -- aucun conflit.  Le drain lit huit mots SWAR en quatre
     * LDS.64 au lieu de huit LDS.32.                                        */
    __shared__ uint32_t sT[NL][6];     /* Q(e) empaquete SWAR, biais 64 */
    __shared__ uint32_t sF[NL][6];     /* part e_lo des ecarts IMPAIRS, biais 64 */
    __shared__ uint32_t sB[NS*WP];     /* bitmaps par (lag, valeur)     */
    __shared__ uint32_t sPw[256][10];  /* [0..3] P(o), [4] o, [6..9] base impaire */
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

    /* --- table sF : la part e_lo des ecarts impairs ------------------------
     * A_{2m+1}(o,e) = Base_m(o,e_hi) + delta_m(o_fixe,e_lo) + 4.sum(o_a & e_c)
     * Le terme delta ne lit que des bits de o COMMUNS a tout le bloc : le
     * mapping u -> o passe par __brev, donc threadIdx ne pilote que les bits
     * [N-9,N-2] de o et tout le reste est constant sur le bloc.  delta ne
     * depend pas non plus de e_hi : la table se construit UNE FOIS par bloc,
     * pas une fois par vhi.  Les rares termes ou le partenaire tombe dans
     * [N-9,N-2] sont laisses au drain (trois a n=31, cf. oe_fix).            */
    for (int el = tid; el < NL; el += 256) {
        const uint32_t eb = (uint32_t)el << 1;      /* e_1..e_K aux bits 1..K */
        uint32_t w[OW];
        #pragma unroll
        for (int k=0;k<OW;k++) w[k]=0x40404040u;
        #pragma unroll
        for (int m=1;m<=MO;m++){
            int d=0;
            #pragma unroll
            for (int c=1;c<=K;c++){
                const int ec=(int)((eb>>c)&1u);
                if (c>=m){     const int a=c-m;
                    if (a>=N-9 && a<=N-2) d -= 2*ec;   /* reporte au drain */
                    else                  d -= 2*ec*(1-2*(int)((o>>a)&1u)); }
                if (c<=N-m-2){ const int a=c+m+1;
                    if (a>=N-9 && a<=N-2) d -= 2*ec;   /* reporte au drain */
                    else                  d -= 2*ec*(1-2*(int)((o>>a)&1u)); }
            }
            const int l=m-1;
            w[l>>2] = (w[l>>2] & ~(0xFFu<<(8*(l&3)))) | ((uint32_t)((d+64)&0xFF)<<(8*(l&3)));
        }
        {   const int l=MO, pv=64+__popc((uint32_t)el);   /* slot de signe */
            w[l>>2] = (w[l>>2] & ~(0xFFu<<(8*(l&3)))) | ((uint32_t)(pv&0xFF)<<(8*(l&3))); }
        #pragma unroll
        for (int k=0;k<OW;k++) sF[el][k]=w[k];
    }

    uint32_t acc[5] = {0,0,0,0,0};
    const int warp = tid>>5, lane = tid&31;
    const unsigned ltm = (1u<<lane)-1u;
    int head = 0, qlen = 0;

    for (uint32_t vv = 0; vv < vcnt; vv++) {
        const uint32_t vhi = vhi0 + vv;
        /* Base_m = A_{2m+1}(o, e_lo=0) : les 15 ecarts impairs evalues une
         * fois par (thread, e_hi), empaquetes en octets SWAR biais 64.  Les
         * 30 popcounts qui etaient PAR SURVIVANT sont maintenant PAR vhi,
         * donc amortis sur les ~16,5 survivants du thread.  L'octet MO porte
         * popc(o)+popc(e_hi), la moitie thread du signe.                    */
        {   const uint32_t e0 = (vhi<<K)<<1;
            uint32_t w[OW];
            #pragma unroll
            for (int k=0;k<OW;k++) w[k]=0x40404040u;
            #pragma unroll
            for (int m=1;m<=MO;m++){
                const int L1=N-m, L2=N-m-1;
                int A = L1 - 2*__popc((o ^ (e0>>m)) & ((1u<<L1)-1u));
                if (L2>0) A += L2 - 2*__popc((e0 ^ (o>>(m+1))) & ((1u<<L2)-1u));
                const int l=m-1;
                w[l>>2] = (w[l>>2] & ~(0xFFu<<(8*(l&3)))) | ((uint32_t)((A+64)&0xFF)<<(8*(l&3)));
            }
            {   const int l=MO, pv=64+__popc(o)+__popc(e0);
                w[l>>2] = (w[l>>2] & ~(0xFFu<<(8*(l&3)))) | ((uint32_t)(pv&0xFF)<<(8*(l&3))); }
            #pragma unroll
            for (int k=0;k<OW;k++) sPw[tid][6+k]=w[k];
        }
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

/* Les b bits de poids faible sont-ils nuls ? (0 <= b <= 160) */
static int h_low_zero(const u160*x,int b){
    for(int k=0;k<b;k++) if((x->w[k>>5]>>(k&31))&1) return 0;
    return 1; }

/* Valuation 2-adique GARANTIE de toute somme partielle.  Chaque terme est un
 * produit des n facteurs A_i, i = 2..n+1, et A_i = i (mod 2) : les ecarts
 * PAIRS donnent donc chacun un facteur 2, et il y en a floor((n+1)/2).  Toute
 * somme de tels termes est divisible par 2^E(n) -- c'est un auto-test par
 * TACHE, disponible immediatement, la ou la divisibilite par 2^{2n} ne vaut
 * que pour le total assemble.  Mesure : les tranches n=31 deja rendues sortent
 * a v2 >= 24, soit 8 bits de marge sur les 16 garantis.                     */
static int even_gaps(int N){ return (N+1)/2; }
static int part_ok(const u160*x,int N,const char*what){
    if(h_low_zero(x,even_gaps(N))) return 1;
    fprintf(stderr,"*** AUTO-TEST ECHOUE (%s) : somme partielle non divisible "
                   "par 2^%d ***\n",what,even_gaps(N));
    fprintf(stderr,"    tranche corrompue -- a rejouer, ne pas l'agreger\n");
    return 0; }

typedef void (*kern_t)(uint32_t,uint32_t,uint32_t,uint32_t*);
typedef void (*dker_t)(uint32_t*);
#define DINST(NN) case NN: return (dker_t)diag_kernel<NN>;
static dker_t dpick(int N){ switch(N){
    DINST(9)  DINST(10) DINST(11) DINST(12) DINST(13) DINST(14) DINST(15) DINST(16)
    DINST(17) DINST(18) DINST(19) DINST(20) DINST(21) DINST(22) DINST(23) DINST(24)
    DINST(27) DINST(28) DINST(31) default: return NULL; } }
#ifndef K_
#define K_ 7          /* balaye : cf. README 4.5 */
#endif
#define INST(NN) case NN: return (kern_t)oe_kernel<NN,K_>;
static kern_t pick(int N){ switch(N){
    INST(9)  INST(10) INST(11) INST(12) INST(13) INST(14) INST(15) INST(16)
    INST(17) INST(18) INST(19) INST(20) INST(21) INST(22) INST(23) INST(24)
    INST(27) INST(28) INST(31) default: return NULL; } }

int main(int argc,char**argv){
    int N=15; long long from=0,count=-1; int chunk=16, diagonly=0, merge=0, bench=0, dev=-1; const char*mfile=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-n")) N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--from")) from=atoll(argv[++i]);
        else if(!strcmp(argv[i],"--count")) count=atoll(argv[++i]);
        else if(!strcmp(argv[i],"--chunk")) chunk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dev")) dev=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--diag")) diagonly=1;
        else if(!strcmp(argv[i],"--bench")) bench=(i+1<argc&&argv[i+1][0]!='-')?atoi(argv[++i]):64;
        else if(!strcmp(argv[i],"--merge")) { merge=i+1; break; }
        else if(!strcmp(argv[i],"--merge-file")) mfile=argv[++i];
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; } }

    /* --merge : additionne des sommes partielles hexadecimales et conclut.
     * Chaque tranche distribuee (y compris --diag) sort une ligne PART=...   */
    if(merge||mfile){
        u160 t; memset(&t,0,sizeof t); long long nread=0;
        /* --merge-file : indispensable des que le nombre de tranches depasse ce
         * qu'une ligne de commande peut porter (8192 tranches ~ 370 Ko d'argv). */
        if(mfile){
            FILE*fp=fopen(mfile,"r"); if(!fp){ perror(mfile); return 1; }
            char ln[128];
            while(fgets(ln,sizeof ln,fp)){ u160 x;
                if(ln[0]=='\n'||ln[0]==0) continue;
                if(sscanf(ln,"%8x:%8x:%8x:%8x:%8x",&x.w[4],&x.w[3],&x.w[2],&x.w[1],&x.w[0])!=5){
                    fprintf(stderr,"partiel illisible ligne %lld : %s",nread+1,ln); fclose(fp); return 1; }
                if(!part_ok(&x,N,"partielle lue")){
                    fprintf(stderr,"    ligne %lld de %s\n",nread+1,mfile); fclose(fp); return 1; }
                h_add(&t,&x); nread++; }
            fclose(fp);
            fprintf(stderr,"%lld tranches lues depuis %s\n",nread,mfile); }
        for(int i=merge;merge&&i<argc;i++){ u160 x;
            if(sscanf(argv[i],"%8x:%8x:%8x:%8x:%8x",&x.w[4],&x.w[3],&x.w[2],&x.w[1],&x.w[0])!=5){
                fprintf(stderr,"partiel illisible : %s\n",argv[i]); return 1; }
            if(!part_ok(&x,N,"partielle lue")){
                fprintf(stderr,"    argument : %s\n",argv[i]); return 1; }
            h_add(&t,&x); }
        h_shl2(&t);
        /* 2n+1 et non 2n : le total vaut 2^{2n}.V(n) et V(n) = 2.L(2,n) est
         * PAIR, car aucun appariement de Langford n'est son propre miroir
         * (il faudrait 2p = 2n-k pour tout k, impossible des que k est impair).
         * Un bit de controle gratuit de plus, et surtout un V impair -- qui
         * serait tronque en silence par le >>1 ci-dessous -- devient une
         * erreur au lieu d'un resultat faux.                                */
        int ok=1; for(int b=0;b<2*N+1;b++) if((t.w[b>>5]>>(b&31))&1) ok=0;
        if(!ok){ fprintf(stderr,"*** AUTO-TEST ECHOUE : non divisible par 2^%d ***\n",2*N+1);
                 fprintf(stderr,"    (tranche manquante, dupliquee, ou corrompue)\n"); return 1; }
        u160 v; memset(&v,0,sizeof v);
        for(int b=2*N;b<160;b++) if((t.w[b>>5]>>(b&31))&1) v.w[(b-2*N)>>5]|=1u<<((b-2*N)&31);
        printf("n=%d   V(n) = 2*L(2,n) = ",N); print_u160(v); printf("\n");
        uint32_t c=0; u160 h;
        for(int q=4;q>=0;q--){ uint32_t x=v.w[q]; h.w[q]=(x>>1)|(c<<31); c=x&1; }
        printf("n=%d   L(2,%d)         = ",N,N); print_u160(h); printf("\n");
        return 0; }

    if(dev>=0) CHECK(cudaSetDevice(dev));   /* noeud multi-GPU : un worker par carte */

    /* La reduction de symetrie n'est PAS valide pour tout n.  Nier la rangee o
     * multiplie le produit par (-1)^{MO} et le poids (prod x_k) par (-1)^N :
     * la sommande n'est invariante que si MO + N est pair, c'est-a-dire
     * exactement quand N = 0 ou 3 (mod 4) -- precisement les n ou une suite de
     * Langford existe.  Aux autres n le noyau rendait un nombre parfaitement
     * faux et parfaitement credible : -n 9 annoncait 5558 la ou la reponse est
     * 0.  `verify.c` le montre a chaque n de 1 a 16.  On refuse donc. */
    if (N % 4 != 0 && N % 4 != 3){
        fprintf(stderr,
          "n=%d : aucune suite de Langford n'existe (n doit valoir 0 ou 3 mod 4),\n"
          "et la reduction de symetrie de ce noyau y est INVALIDE -- il rendrait\n"
          "un nombre faux sans qu'aucun auto-test ne s'en apercoive.  Voir verify.c\n", N);
        return 1; }
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

    /* --bench [S] : calibre le debit sur CE GPU sans faire tourner le calcul.
     * On tire S valeurs de vhi au hasard mais avec une graine FIXE, donc deux
     * GPU differents mesurent exactement le meme echantillon : la comparaison
     * est appariee et le rapport est bien plus precis que chaque estimation
     * absolue prise separement.  Le cout par vhi varie d'un facteur ~10 selon
     * le motif de bits (elagage par les bitmaps de survie), d'ou la necessite
     * d'un tirage uniforme plutot que de quelques points choisis.            */
    if(bench){
        cudaDeviceProp pr; CHECK(cudaGetDeviceProperties(&pr,0));
        int khz=0; cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,0);
        printf("GPU        : %s  (%d SM, sm_%d%d, %.0f MHz)\n",
               pr.name,pr.multiProcessorCount,pr.major,pr.minor,khz/1000.0);
        /* rodage : monte les horloges avant de chronometrer */
        for(int w=0;w<3;w++){ Kf<<<blocks,256>>>(0u,1u,0u,d_out); }
        CHECK(cudaDeviceSynchronize());
        /* Deux corrections, apres confrontation a une mesure a travail identique
         * sur deux cartes ou l'ancienne version annoncait x5,8 pour un vrai x3,05 :
         *   - splitmix64 remplace le LCG, dont les bits de poids faible donnaient
         *     un sous-ensemble de vhi de plus en plus cher a mesure qu'on tirait
         *     (juste a 64 tirages, 2x trop pessimiste a 128) ;
         *   - on chronometre le chemin de PRODUCTION -- meme count qu'un --chunk,
         *     recopie et reduction hote comprises -- et non un lancement isole a
         *     count=1, qui ne charge pas deux cartes de la meme facon.        */
        uint64_t st=0; double sum=0,sum2=0; int S=bench, got=0;
        const int CB = 4;                    /* vhi par lancement, comme en production */
        struct timespec A,B;
        while(got<S){
            st += 0x9E3779B97F4A7C15ull;                   /* splitmix64 */
            uint64_t z = st;
            z = (z ^ (z>>30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z>>27)) * 0x94D049BB133111EBull;
            z ^= z>>31;
            long long s0 = (long long)(z % (unsigned long long)(nvhi-CB));
            long long b0 = ((long long)s0<<K_)>>8; if(b0<0) b0=0;
            int nb = blocks-(int)b0; if(nb<=0) continue;
            clock_gettime(CLOCK_MONOTONIC,&A);
            Kf<<<nb,256>>>((uint32_t)s0,(uint32_t)CB,(uint32_t)b0,d_out);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(h_out,d_out,(size_t)nb*5*4,cudaMemcpyDeviceToHost));
            { u160 acc; memset(&acc,0,sizeof acc);
              for(int q=0;q<nb;q++) h_add(&acc,(u160*)(h_out+5*q)); }
            clock_gettime(CLOCK_MONOTONIC,&B);
            double t=((B.tv_sec-A.tv_sec)+(B.tv_nsec-A.tv_nsec)*1e-9)/CB;
            sum+=t; sum2+=t*t; got++;
        }
        double m=sum/S, var=(sum2-S*m*m)/(S-1), se=sqrt(var/S);
        double tot=m*(double)nvhi;
        printf("echantillon: %d vhi (graine fixe)  moyenne %.4f s/vhi  err-std %.1f%%\n",
               S,m,100*se/m);
        printf("PROJECTION n=%d : %.1f h GPU = %.2f jours   (IC95%% %.1f .. %.1f h)\n",
               N, tot/3600.0, tot/86400.0, (m-2*se)*nvhi/3600.0, (m+2*se)*nvhi/3600.0);
        printf("BENCH_SPV=%.6f  BENCH_GPUH=%.2f\n", m, tot/3600.0);
        return 0; }

    u160 total; memset(&total,0,sizeof total);
    struct timespec T0,T1; clock_gettime(CLOCK_MONOTONIC,&T0);
    long long done=0;
    if(diagonly){                       /* uniquement les orbites fixes e=f(o) */
        dker_t Dk=dpick(N); Dk<<<blocks,256>>>(d_out); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_out,d_out,(size_t)blocks*5*4,cudaMemcpyDeviceToHost));
        for(int q=0;q<blocks;q++) h_add(&total,(u160*)(h_out+5*q));
        if(!part_ok(&total,N,"diagonale")) return 1;
        printf("PART=%08x:%08x:%08x:%08x:%08x   (diagonale n=%d)\n",
               total.w[4],total.w[3],total.w[2],total.w[1],total.w[0],N);
        return 0; }
    for(long long s=from;s<from+count;s+=chunk){
        int c=(int)((s+chunk<=from+count)?chunk:(from+count-s));
        /* seuls les blocs avec u >= v peuvent contribuer (predicat v <= u) */
        long long b0 = ((long long)s<<K_)>>8; if(b0<0) b0=0;
        int nb = blocks-(int)b0; if(nb<=0){ done+=c; continue; }
        Kf<<<nb,256>>>((uint32_t)s,(uint32_t)c,(uint32_t)b0,d_out);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_out,d_out,(size_t)nb*5*4,cudaMemcpyDeviceToHost));
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
        /* 2n+1 : cf. --merge, V(n) = 2.L(2,n) est pair. */
        int ok=1; for(int b=0;b<2*N+1;b++) if((total.w[b>>5]>>(b&31))&1) ok=0;
        if(!ok){ fprintf(stderr,"*** AUTO-TEST ECHOUE : non divisible par 2^%d ***\n",2*N+1); return 1; }
        u160 v; memset(&v,0,sizeof v);
        for(int b=2*N;b<160;b++) if((total.w[b>>5]>>(b&31))&1) v.w[(b-2*N)>>5]|=1u<<((b-2*N)&31);
        printf("n=%d   V(n) = 2*L(2,n) = ",N); print_u160(v); printf("\n");
        uint32_t c=0; u160 h;
        for(int q=4;q>=0;q--){ uint32_t x=v.w[q]; h.w[q]=(x>>1)|(c<<31); c=x&1; }
        printf("n=%d   L(2,%d)         = ",N,N); print_u160(h); printf("\n");
    } else {
        /* auto-test par tache : la tranche est rejetee ici, sur la machine qui
         * l'a produite, au lieu d'etre decouverte a l'agregation finale.     */
        if(!part_ok(&total,N,"tranche")) return 1;
        printf("PART=%08x:%08x:%08x:%08x:%08x   (n=%d vhi %lld..%lld)\n",
               total.w[4],total.w[3],total.w[2],total.w[1],total.w[0],N,from,from+done-1);
    }
    return 0;
}
