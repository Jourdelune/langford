/* Unbiased Monte-Carlo estimate of L(2,n)  (Knuth's 1975 tree-size estimator).
 *
 * Walk one random root-to-leaf path.  At each node let c = number of legal
 * children; pick one uniformly and multiply the running weight by c.  The
 * expectation of the weight is exactly the number of leaves, i.e. the number
 * of Langford pairings.  Averaging independent walks converges to L(2,n)*2.
 * Values are branched most-constrained-first, which cuts the variance a lot.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

static inline uint64_t xs(uint64_t *s){ uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

int main(int argc,char**argv){
    int n = argc>1?atoi(argv[1]):31;
    long long trials = argc>2?atoll(argv[2]):2000000;
    int M = 2*n;
    double sum = 0, sumsq = 0; long long done = 0;

    #pragma omp parallel reduction(+:sum,sumsq,done)
    {
        uint64_t seed = 88172645463325252ULL ^ (uint64_t)(omp_get_thread_num()*2654435761u+12345);
        for(int w=0;w<64;w++) xs(&seed);
        int optP[64]; int used[64];
        #pragma omp for schedule(static)
        for(long long it=0; it<trials; it++){
            uint64_t occ = 0; double weight = 1.0; int left = n;
            for(int v=1;v<=n;v++) used[v]=0;
            while(left>0){
                /* most constrained unplaced value */
                int bestv=-1, bestc=1<<30;
                for(int v=1;v<=n;v++){
                    if(used[v]) continue;
                    int c=0;
                    for(int p=0;p+v+1<M;p++){
                        uint64_t m=(1ULL<<p)|(1ULL<<(p+v+1));
                        if(!(occ&m)) c++;
                    }
                    if(c<bestc){ bestc=c; bestv=v; if(!c) break; }
                }
                if(bestc==0){ weight=0; break; }
                int pick = (int)(xs(&seed) % (uint64_t)bestc), c=0, chosen=-1;
                for(int p=0;p+bestv+1<M;p++){
                    uint64_t m=(1ULL<<p)|(1ULL<<(p+bestv+1));
                    if(!(occ&m)){ if(c==pick){ chosen=p; break; } c++; }
                }
                occ |= (1ULL<<chosen)|(1ULL<<(chosen+bestv+1));
                used[bestv]=1; left--; weight *= (double)bestc;
            }
            sum += weight; sumsq += weight*weight; done++;
        }
    }
    double mean = sum/done;
    double var  = (sumsq/done - mean*mean)/done;
    double sd   = sqrt(var>0?var:0);
    printf("n=%d  trials=%lld\n", n, done);
    printf("  E[#sequences] = %.6g   (L(2,%d) = %.6g)\n", mean, n, mean/2);
    printf("  std err       = %.3g  (%.2f%%)\n", sd, 100.0*sd/mean);
    printf("  log2(V)       = %.3f\n", log2(mean));
    return 0;
}
