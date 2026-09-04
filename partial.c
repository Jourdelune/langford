/* GRASSMANNISATION PARTIELLE.
 *
 * Godfrey extrait [x_1..x_2n] prod A_i avec des variables COMMUTANTES : 2^{2n}.
 * Le Pfaffien utilise des variables de Grassmann ANTICOMMUTANTES : 2^n.poly,
 * mais le resultat est signe et le signe n'est pas rattrapable (pfaff_test).
 *
 * Rien n'oblige a choisir. Soit A un sous-ensemble de positions rendues
 * grassmanniennes, le reste restant commutant. Le coefficient s'extrait alors
 * en 2^{|A^c|} evaluations +-1 (l'homogeneite tient : degre y = |A^c| = nombre
 * de variables y) fois 2^n pour les couleurs, soit 2^{n+|A^c|}.
 *   |A| = 2n  -> 2^n   mais signe (Kasteleyn, mort)
 *   |A| = 0   -> 2^{2n} = Godfrey
 *   |A| > n   -> SOUS le 4^n.
 * Il faut que sign_A(M) . prod_{e in M} w_e soit constant sur les appariements
 * de Langford. Ce programme cherche le plus grand A pour lequel c'est possible.
 *
 * sign_A(M) : on lit les positions de A dans l'ordre des couleurs (2,3,...,n+1),
 * et pour chaque couleur son ouverture puis sa fermeture ; sign = parite du
 * nombre d'inversions de la suite obtenue.                                    */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int N,M2,NE,NW;
static int eoff[64];
static int A_[40],B_[40],nc;
static uint64_t occ,used;
static uint64_t Amask;           /* positions grassmanniennes */
static uint64_t *basis; static int *has; static uint64_t *row;
static long long bad, nm;

static inline void setbit(uint64_t*v,int i){ v[i>>6]^=1ULL<<(i&63); }

static void feed(void)
{
    /* signe restreint a A : parite des inversions de la suite lue en ordre couleur */
    int seq[80], ns=0;
    for(int c=0;c<nc;c++){
        if((Amask>>A_[c])&1) seq[ns++]=A_[c];
        if((Amask>>B_[c])&1) seq[ns++]=B_[c];
    }
    int inv=0;
    for(int i=0;i<ns;i++) for(int j=i+1;j<ns;j++) if(seq[i]>seq[j]) inv++;

    static uint64_t r[64];
    memcpy(r,row,NW*8);
    setbit(r,NE);                       /* la constante */
    if(inv&1) setbit(r,NE+1);           /* second membre */
    nm++;
    for(int p=0;p<=NE;p++){
        if(!((r[p>>6]>>(p&63))&1)) continue;
        if(has[p]){ for(int k=0;k<NW;k++) r[k]^=basis[(size_t)p*NW+k]; }
        else { memcpy(basis+(size_t)p*NW,r,NW*8); has[p]=1; return; }
    }
    if((r[(NE+1)>>6]>>((NE+1)&63))&1) bad++;
}

static void rec(void){
    int p=0; while(p<M2 && ((occ>>p)&1)) p++;
    if(p==M2){ feed(); return; }
    for(int d=2;d<=N+1;d++){
        if((used>>d)&1) continue;
        int q=p+d; if(q>=M2||((occ>>q)&1)) continue;
        int ei=eoff[d]+p;
        A_[nc]=p; B_[nc]=q; nc++;
        setbit(row,ei); used|=1ULL<<d; occ|=(1ULL<<p)|(1ULL<<q);
        rec();
        occ&=~((1ULL<<p)|(1ULL<<q)); used&=~(1ULL<<d); setbit(row,ei); nc--;
    }
}

static int consistent(uint64_t A){
    Amask=A; bad=0; nm=0;
    memset(basis,0,(size_t)(NE+1)*NW*8); memset(has,0,(NE+1)*4);
    memset(row,0,NW*8); occ=used=0; nc=0;
    rec();
    return bad==0;
}

int main(int argc,char**argv){
    N=atoi(argv[1]); M2=2*N;
    NE=0; for(int d=2;d<=N+1;d++){ eoff[d]=NE; NE+=M2-d; }
    NW=(NE+2+63)/64;
    basis=malloc((size_t)(NE+1)*NW*8); has=malloc((NE+1)*4); row=malloc(NW*8);

    /* glouton : on ajoute des positions a A tant que le systeme reste soluble */
    uint64_t bestA=0; int sz=0;
    unsigned st=12345;
    for(int trial=0; trial<40; trial++){
        int order[64]; for(int i=0;i<M2;i++) order[i]=i;
        if(trial){ for(int i=M2-1;i>0;i--){ st=st*1103515245u+12345u; int j=(st>>16)%(i+1);
                    int t=order[i];order[i]=order[j];order[j]=t; } }
        uint64_t A=0; int s2=0, improved=1;
        while(improved){ improved=0;
            for(int i=0;i<M2;i++){
                if((A>>order[i])&1) continue;
                uint64_t A2=A|(1ULL<<order[i]);
                if(consistent(A2)){ A=A2; s2++; improved=1; } } }
        if(s2>sz){ sz=s2; bestA=A; }
    }
    uint64_t A=bestA;
    printf("n=%2d  2n=%2d  |A| max (glouton) = %2d   -> cout 2^{n+|A^c|} = 2^{%d}   contre 4^n = 2^%d   %s\n",
           N,M2,sz,N+(M2-sz),2*N, (N+(M2-sz) < 2*N) ? "GAIN" : "pas de gain");
    printf("      A = ");
    for(int i=0;i<M2;i++) if((A>>i)&1) printf("%d ",i+1);
    printf("\n");
    return 0;
}
