/* Le contrainte "les n ecarts sont exactement {2,...,n+1}" se ramene a UNE
 * SEULE equation entiere.
 *
 * Soit m_i la multiplicite de la couleur i dans le multi-ensemble des ecarts.
 * On a toujours sum_i m_i = n (n cordes). Je pretends :
 *
 *      sum_j 2^{d_j} = 2^{n+2} - 4   <=>   tous les m_i valent 1
 *
 * Preuve : parmi toutes les ecritures de N comme sum m_i 2^i avec m_i >= 0, la
 * representation BINAIRE minimise la somme des chiffres (chaque report
 * 2^{i+1} -> 2 x 2^i augmente la somme des chiffres de 1). Or 2^{n+2}-4 =
 * 2^2+2^3+...+2^{n+1} a exactement n chiffres a 1. Donc l'ecriture de somme de
 * chiffres n est unique : c'est la binaire, tous les m_i = 1.
 *
 * Verification exhaustive sur tous les multi-ensembles.                       */
#include <stdio.h>
#include <stdlib.h>
static int N; static long long bad=0, tot=0;
static int m[40];
static void go(int i,int left,long long sum){
    if(i>N+1){ if(left) return; tot++;
        long long T=((long long)1<<(N+2))-4;
        if(sum==T){ int ok=1; for(int k=2;k<=N+1;k++) if(m[k]!=1) ok=0;
                    if(!ok){ bad++; if(bad<3){ printf("   CONTRE-EXEMPLE : ");
                        for(int k=2;k<=N+1;k++) if(m[k]) printf("%d^%d ",k,m[k]); printf("\n"); } } }
        return; }
    for(int c=0;c<=left;c++){ m[i]=c; go(i+1,left-c,sum+(long long)c*((long long)1<<i)); }
    m[i]=0;
}
int main(int argc,char**argv){ N=atoi(argv[1]); go(2,N,0);
    printf("n=%2d  multi-ensembles testes=%lld  contre-exemples=%lld  -> %s\n",
           N,tot,bad,bad?"FAUX":"vrai");
    return 0; }
