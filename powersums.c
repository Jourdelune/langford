/* Si la contrainte de couleurs se ramene a peu de SOMMES DE PUISSANCES, on gagne.
 *
 * Imposer p_j = sum_l d_l^j pour j=1..k par transformee de Fourier coute
 * prod_j (portee de p_j) ~ n^{k(k+3)/2}, et le comptage pondere des couplages
 * coute 2^n. Donc le total est n^{k(k+3)/2} . 2^n, a comparer a 4^n = 2^{2n} :
 *
 *    k=1 : 31^2 . 2^31 = 2^40,9      k=2 : 31^5 . 2^31 = 2^55,8   <-- GAIN
 *    k=3 : 31^9 . 2^31 = 2^75,6      (perdu)
 *
 * Reste a savoir combien de multi-ensembles partagent les k premieres sommes de
 * puissances avec l'ensemble plein {2,...,n+1}. Ce programme les compte.      */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int N,K; static long long cnt[8];
static long long tgt[8];
static int m[40];
static void go(int i,int left,long long ps[8]){
    if(i>N+1){ if(left) return;
        for(int k=1;k<=K;k++){ if(ps[k]!=tgt[k]) return; cnt[k]++; }
        return; }
    for(int c=0;c<=left;c++){
        long long np[8]; memcpy(np,ps,sizeof np);
        long long pw=1;
        for(int k=1;k<=K;k++){ pw*= i; np[k]=ps[k]+(long long)c*pw; }
        m[i]=c; go(i+1,left-c,np); }
    m[i]=0;
}
int main(int argc,char**argv){
    N=atoi(argv[1]); K=5;
    memset(tgt,0,sizeof tgt);
    for(int i=2;i<=N+1;i++){ long long pw=1; for(int k=1;k<=K;k++){ pw*=i; tgt[k]+=pw; } }
    long long ps[8]={0}; memset(cnt,0,sizeof cnt);
    go(2,N,ps);
    printf("n=%2d : multi-ensembles partageant p_1..p_k avec l'ensemble plein :",N);
    for(int k=1;k<=K;k++) printf("  k=%d:%lld",k,cnt[k]);
    printf("\n");
    return 0; }
