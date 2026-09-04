/* IDEE : le compte arc-en-ciel vaut  sum_S (-1)^{n-|S|} haf(A_S), et les A_S
 * sont des matrices de TOEPLITZ (A[p][q] ne depend que de p-q).  Sur un chemin,
 * la matrice de transfert est la MEME a chaque position, donc la suite
 *      N_S(k) = #couplages parfaits du graphe des ecarts S sur [1..k]
 * satisfait une recurrence lineaire.  Si son ordre r est petit, on calcule r
 * termes pour de petits k (ou la DP est bon marche) et on extrapole jusqu'a
 * k=2n : haf(A_S) en poly, donc le tout en 2^n.poly.  Ce test mesure r.      */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef unsigned long long u64;
static const u64 P = 2147483647ULL;          /* 2^31-1 */
static u64 ad(u64 a,u64 b){ a+=b; return a>=P?a-P:a; }
static u64 sb(u64 a,u64 b){ return a>=b?a-b:a+P-b; }
static u64 ml(u64 a,u64 b){ return (a*b)%P; }
static u64 pw(u64 a,u64 e){ u64 r=1; while(e){ if(e&1) r=ml(r,a); a=ml(a,a); e>>=1;} return r; }
static u64 iv(u64 a){ return pw(a,P-2); }

/* #couplages parfaits sur [0..k-1], aretes {p,p+d} pour d dans S */
static u64 countPM(int k, const int*S, int ns, int m)
{
    int W = m+1;                              /* bit j = position p+j reservee */
    size_t NS = (size_t)1<<W;
    u64 *cur = calloc(NS,8), *nx = calloc(NS,8);
    cur[0]=1;
    for(int p=0;p<k;p++){
        memset(nx,0,NS*8);
        for(size_t st=0; st<NS; st++){
            u64 v=cur[st]; if(!v) continue;
            if(st&1){ nx[st>>1]=ad(nx[st>>1],v); }      /* position reservee */
            else {
                for(int j=0;j<ns;j++){ int d=S[j];
                    if(p+d>=k) continue;
                    size_t bit=(size_t)1<<d;
                    if(st&bit) continue;
                    size_t s2=(st|bit)>>1;
                    nx[s2]=ad(nx[s2],v);
                }
            }
        }
        u64*t=cur; cur=nx; nx=t;
    }
    u64 r=cur[0]; free(cur); free(nx); return r;
}

/* Berlekamp-Massey : ordre minimal de la recurrence lineaire */
static int bm(const u64*s,int n)
{
    u64 *C=calloc(n+1,8), *B=calloc(n+1,8), *T=calloc(n+1,8);
    C[0]=1; B[0]=1; int L=0,mm=1; u64 b=1;
    for(int i=0;i<n;i++){
        u64 d=s[i];
        for(int j=1;j<=L;j++) d=ad(d,ml(C[j],s[i-j]));
        if(d==0){ mm++; }
        else if(2*L<=i){
            memcpy(T,C,(n+1)*8);
            u64 c=ml(d,iv(b));
            for(int j=0;j+mm<=n;j++) C[j+mm]=sb(C[j+mm],ml(c,B[j]));
            L=i+1-L; memcpy(B,T,(n+1)*8); b=d; mm=1;
        } else {
            u64 c=ml(d,iv(b));
            for(int j=0;j+mm<=n;j++) C[j+mm]=sb(C[j+mm],ml(c,B[j]));
            mm++;
        }
    }
    free(C);free(B);free(T); return L;
}

int main(int argc,char**argv)
{
    /* familles S de test : intervalles {2..m} de taille croissante */
    for(int m=3; m<=12; m++){
        int S[16], ns=0;
        for(int d=2; d<=m; d++) S[ns++]=d;
        int K = 4*(1<<(m<10?m:10)); if(K>1400) K=1400;
        int nt = K;
        u64 *seq=malloc(nt*8);
        for(int k=0;k<nt;k++) seq[k]=countPM(k,S,ns,m);
        int r=bm(seq,nt);
        printf("S={2..%2d}  |S|=%2d  fenetre=2^%d=%5d etats   ordre de la recurrence = %d",
               m,ns,m,1<<m,r);
        if(r*2+2>nt) printf("   (peut etre tronque : %d termes)",nt);
        printf("\n");
        free(seq);
    }
    return 0;
}
