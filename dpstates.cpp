/* Exact number of REACHABLE (window, used-colours) states of the position DP,
 * cut by cut.  README §3.4 used sum_b C(n,b)*C(n,(n+b)/2) as the count, which
 * ignores that every reserved position must be backed by a distinct unused
 * colour >= its distance (a Hall condition).  Measure the truth and fit.     */
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
int main(int argc,char**argv){
    int N=atoi(argv[1]); int M=2*N;
    std::vector<uint64_t> cur, nxt;
    cur.push_back(0);                       /* cut 0: nothing reserved, no colours */
    double peak=0; double total=0, trans=0; int peakcut=0;
    for(int p=0;p<M;p++){
        nxt.clear();
        for(uint64_t st : cur){
            uint32_t R=(uint32_t)(st&0xFFFFFFFFu), C=(uint32_t)(st>>32);
            if(R&1u){ nxt.push_back((uint64_t)(R>>1) | ((uint64_t)C<<32)); trans+=1; }
            else {
                for(int i=2;i<=N+1;i++){
                    if(C&(1u<<i)) continue;
                    if(p+1+i>M) continue;
                    if(R&(1u<<i)) continue;
                    uint32_t R2=(R|(1u<<i))>>1, C2=C|(1u<<i);
                    nxt.push_back((uint64_t)R2 | ((uint64_t)C2<<32)); trans+=1;
                }
            }
        }
        std::sort(nxt.begin(),nxt.end());
        nxt.erase(std::unique(nxt.begin(),nxt.end()),nxt.end());
        cur.swap(nxt);
        total+=cur.size();
        if(cur.size()>peak){ peak=cur.size(); peakcut=p+1; }
    }
    printf("n=%2d  peak states=%.6g (cut %d)  total states=%.6g  transitions=%.6g   log2(peak)=%.2f  log2(peak)/n=%.3f\n",
           N,peak,peakcut,total,trans, peak>0?log2(peak):0, peak>0?log2(peak)/N:0);
    return 0;
}
