/* Deux faits structurels que je derive ici, et leur consequence algorithmique.
 *
 * (1) Pour toute corde c de longueur d dans un diagramme de cordes :
 *        #croisements(c) = d - 1 - 2*#{cordes entierement interieures a c}
 *     donc  #croisements(c) = d - 1  (mod 2).   -- LOCAL, pas global.
 *     Consequence :  2*cr = sum_c #crois(c)  donne  cr = nest + (sum_c (d-1))/2,
 *     ou nest = nombre de paires imbriquees.  Donc cr = nest + const (mod 2).
 *
 * (2) Generalisation de Kasteleyn : au lieu d'exiger que cr soit AFFINE en
 *     l'ensemble d'aretes (ce qui est refute par pfaff_test), il suffirait que
 *     cr se factorise par d formes lineaires -- un Pfaffien sur l'algebre de
 *     groupe (Z/2)^d, de dimension 2^d, donnerait alors 2^{n+d}.poly.
 *     Le test : d ~ log2(#vecteurs-differences "mauvais" distincts).          */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static int N,M2;
static int a_[40],b_[40],nc;
static long long tot=0, badlocal=0, crne=0;
static long long crpar_sum=0;
static uint64_t *diffs=NULL; static long long ndiff=0, capd=0;
static uint64_t occ,used;

static int cmp64(const void*x,const void*y){ uint64_t p=*(const uint64_t*)x,q=*(const uint64_t*)y;
    return p<q?-1:p>q?1:0; }

static void record(void){
    int cr=0, nest=0;
    for(int i=0;i<nc;i++){
        int cross_i=0, inside_i=0;
        for(int j=0;j<nc;j++){ if(i==j) continue;
            int a=a_[i],b=b_[i],c=a_[j],d=b_[j];
            if(a<c&&c<b&&b<d) cross_i++;
            else if(c<a&&a<d&&d<b) cross_i++;
            else if(a<c&&d<b) inside_i++;
        }
        /* fait (1) : cross_i doit valoir (b_i - a_i) - 1 - 2*inside_i */
        if(cross_i != (b_[i]-a_[i]) - 1 - 2*inside_i) badlocal++;
        cr+=cross_i; nest+=inside_i;
    }
    cr/=2;
    crpar_sum += (cr&1);
    /* fait (1bis) : cr = nest + const (mod 2) */
    static int c0=-1; int k=(cr-nest)&1; if(c0<0) c0=k; else if(k!=c0) crne++;
    tot++;
    /* empreinte de l'ensemble d'aretes, pour le test (2) */
    uint64_t h=1469598103934665603ULL;
    for(int i=0;i<nc;i++){ h=(h^(uint64_t)(a_[i]*64+b_[i]))*1099511628211ULL; }
    if(ndiff==capd){ capd=capd?capd*2:1024; diffs=realloc(diffs,capd*8); }
    diffs[ndiff++] = (h<<1) | (uint64_t)(cr&1);
}
static void rec(void){
    int p=0; while(p<M2 && ((occ>>p)&1)) p++;
    if(p==M2){ record(); return; }
    for(int d=2;d<=N+1;d++){
        if((used>>d)&1) continue;
        int q=p+d; if(q>=M2||((occ>>q)&1)) continue;
        a_[nc]=p; b_[nc]=q; nc++;
        used|=1ULL<<d; occ|=(1ULL<<p)|(1ULL<<q);
        rec();
        occ&=~((1ULL<<p)|(1ULL<<q)); used&=~(1ULL<<d); nc--;
    }
}
int main(int argc,char**argv){
    N=atoi(argv[1]); M2=2*N; occ=used=0; nc=0;
    rec();
    printf("n=%2d  appariements=%lld   fait(1) violations=%lld   fait(1bis) violations=%lld   cr impair=%lld\n",
           N,tot,badlocal,crne,crpar_sum);
    /* (2) : borne inferieure sur d, via le nombre de paires a cr different */
    long long odd=crpar_sum, even=tot-odd;
    double npairs=(double)odd*(double)even;
    printf("      paires a cr different = %.6g  ->  d >= log2 = %.1f   (algo en 2^{n+d} = 2^{%.1f} contre 4^n = 2^%d)\n",
           npairs, npairs>0?log2(npairs):0.0, N+(npairs>0?log2(npairs):0.0), 2*N);
    free(diffs); return 0;
}
