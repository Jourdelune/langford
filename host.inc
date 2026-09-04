/* ====================== host side ====================== */
typedef struct { uint32_t w[5]; } u160;
static void h_add(u160 *z, const u160 *x){
    uint64_t c = 0;
    for (int k = 0; k < 5; k++){ uint64_t s = (uint64_t)z->w[k] + x->w[k] + c; z->w[k]=(uint32_t)s; c = s>>32; }
}
static void h_shl2(u160 *z){
    uint32_t carry = 0;
    for (int k = 0; k < 5; k++){ uint32_t v = z->w[k]; z->w[k] = (v<<2)|carry; carry = v>>30; }
}
static void print_u160(u160 v){
    char out[64]; int len = 0; uint32_t t[5]; memcpy(t, v.w, sizeof t);
    for (;;){
        uint64_t rem = 0;
        for (int k = 4; k >= 0; k--){ uint64_t cur = (rem<<32) | t[k]; t[k]=(uint32_t)(cur/1000000000ULL); rem = cur%1000000000ULL; }
        int nz = 0; for (int k = 0; k < 5; k++) if (t[k]) nz = 1;
        for (int d = 0; d < 9; d++){ out[len++] = '0' + (int)(rem%10); rem/=10; }
        if (!nz) break;
    }
    while (len > 1 && out[len-1]=='0') len--;
    for (int k = len-1; k >= 0; k--) putchar(out[k]);
}
static void print_i160(u160 v){                 /* two's complement, signed */
    if (v.w[4] & 0x80000000u){
        putchar('-');
        uint64_t c = 1;
        for (int k = 0; k < 5; k++){ uint64_t s = (uint64_t)(~v.w[k]) + c; v.w[k]=(uint32_t)s; c = s>>32; }
    }
    print_u160(v);
}

typedef void (*kern_t)(uint64_t, int, uint32_t*);
static kern_t pick(int N){
    switch(N){
      case 7:  return godfrey<7>;   case 8:  return godfrey<8>;
      case 11: return godfrey<11>;  case 12: return godfrey<12>;
      case 15: return godfrey<15>;  case 16: return godfrey<16>;
      case 19: return godfrey<19>;  case 20: return godfrey<20>;
      case 23: return godfrey<23>;  case 24: return godfrey<24>;
      case 27: return godfrey<27>;  case 28: return godfrey<28>;
      case 31: return godfrey<31>;  case 32: return godfrey<32>;
      default: return NULL;
    }
}
static void build_tables(int N, int innerBits){
    const int M = 2*N;
    uint32_t hA[32]={0}, hB[32]={0}, hSub[32][8]; int hSh[32]={0};
    memset(hSub, 0, sizeof hSub);
    for (int j = 0; j < innerBits; j++){
        int ca = M - 2 - j;  if (ca > N) ca = N;  if (ca < 0) ca = 0;
        int cb = j - 1;      if (cb > N) cb = N;  if (cb < 0) cb = 0;
        hA[j] = (ca >= 32) ? 0xFFFFFFFFu : ((1u << ca) - 1);
        hB[j] = (cb >= 32) ? 0xFFFFFFFFu : ((1u << cb) - 1);
        hSh[j] = (j >= 2) ? (M + 1 - j) : 0;
        for (int l = 0; l < 32; l++){
            uint32_t v = 2*((hA[j]>>l)&1) + 2*((hB[j]>>l)&1);
            hSub[j][l>>2] |= v << (8*(l&3));
        }
    }
    CHECK(cudaMemcpyToSymbol(cValidA, hA, sizeof hA));
    CHECK(cudaMemcpyToSymbol(cValidB, hB, sizeof hB));
    CHECK(cudaMemcpyToSymbol(cSub,  hSub, sizeof hSub));
    CHECK(cudaMemcpyToSymbol(cShiftB, hSh, sizeof hSh));
}

/* ---- final arithmetic: raw sum over 2^{2n-2} vectors  ->  L(2,n) ---- */
static int finish(int N, u160 total){
    const int M = 2*N;
    h_shl2(&total);                                  /* Z2 x Z2 symmetry: x4 */
    int ok = 1;
    for (int bit = 0; bit < M; bit++) if ((total.w[bit>>5] >> (bit&31)) & 1) ok = 0;
    if (!ok){ fprintf(stderr,"*** SELF-CHECK FAILED: sum not divisible by 2^%d ***\n", M); return 1; }
    u160 v; memset(&v,0,sizeof v);
    for (int bit = M; bit < 160; bit++)
        if ((total.w[bit>>5]>>(bit&31)) & 1) v.w[(bit-M)>>5] |= 1u << ((bit-M)&31);
    printf("n=%d   V(n) = 2*L(2,n) = ", N); print_u160(v); printf("\n");
    uint32_t carry = 0; u160 half;
    for (int k = 4; k >= 0; k--){ uint32_t x = v.w[k]; half.w[k]=(x>>1)|(carry<<31); carry = x&1; }
    printf("n=%d   L(2,%d)         = ", N, N); print_u160(half); printf("\n");
    return 0;
}

/* ---------------- partial-result files (distributed runs) ------------- */
static void write_part(const char *path, int N, int chunkLog, long long from,
                       long long count, u160 s, double secs){
    FILE *f = fopen(path,"w");
    if (!f){ perror("write_part"); return; }
    fprintf(f,"langford-partial 1\nn %d\nchunklog %d\nfrom %lld\ncount %lld\n"
              "sum %08x %08x %08x %08x %08x\nseconds %.3f\n",
            N, chunkLog, from, count, s.w[0],s.w[1],s.w[2],s.w[3],s.w[4], secs);
    fclose(f);
}

static int do_merge(int argc, char **argv, int first){
    int N = -1, chunkLog = -1;
    long long totalChunks = 0;
    static unsigned char *seen = NULL;
    u160 total; memset(&total,0,sizeof total);
    for (int a = first; a < argc; a++){
        FILE *f = fopen(argv[a],"r");
        if (!f){ fprintf(stderr,"cannot open %s\n", argv[a]); return 1; }
        int n2, cl; long long fr, ct; u160 s; double secs; char magic[64]; int ver;
        if (fscanf(f,"%63s %d",magic,&ver)!=2 || strcmp(magic,"langford-partial")){
            fprintf(stderr,"%s: not a partial file\n",argv[a]); return 1; }
        if (fscanf(f," n %d chunklog %d from %lld count %lld sum %x %x %x %x %x seconds %lf",
                   &n2,&cl,&fr,&ct,&s.w[0],&s.w[1],&s.w[2],&s.w[3],&s.w[4],&secs)!=10){
            fprintf(stderr,"%s: parse error\n",argv[a]); return 1; }
        fclose(f);
        if (N < 0){ N = n2; chunkLog = cl; totalChunks = 1LL << (2*N-2-chunkLog);
                    seen = (unsigned char*)calloc(totalChunks,1); }
        if (n2 != N || cl != chunkLog){ fprintf(stderr,"%s: mismatched n/chunklog\n",argv[a]); return 1; }
        for (long long c = fr; c < fr+ct; c++){
            if (c < 0 || c >= totalChunks){ fprintf(stderr,"%s: chunk %lld out of range\n",argv[a],c); return 1; }
            if (seen[c]){ fprintf(stderr,"%s: chunk %lld counted twice\n",argv[a],c); return 1; }
            seen[c] = 1;
        }
        h_add(&total,&s);
        fprintf(stderr,"  + %s : chunks %lld..%lld\n", argv[a], fr, fr+ct-1);
    }
    long long missing = 0;
    for (long long c = 0; c < totalChunks; c++) if (!seen[c]) missing++;
    if (missing){ fprintf(stderr,"INCOMPLETE: %lld of %lld chunks missing\n", missing, totalChunks);
                  printf("partial sum = "); print_i160(total); printf("\n"); return 1; }
    fprintf(stderr,"all %lld chunks present\n", totalChunks);
    return finish(N, total);
}

int main(int argc, char **argv){
    int N = 12, innerBits = 14, chunkLog = -1;
    long long from = 0, count = -1;
    const char *ckpt = NULL, *part = NULL;
    for (int i = 1; i < argc; i++){
        if (!strcmp(argv[i],"--merge"))         return do_merge(argc, argv, i+1);
        else if (!strcmp(argv[i],"-n"))         N = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--inner"))    innerBits = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--chunklog")) chunkLog = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--from"))     from = atoll(argv[++i]);
        else if (!strcmp(argv[i],"--count"))    count = atoll(argv[++i]);
        else if (!strcmp(argv[i],"--ckpt"))     ckpt = argv[++i];
        else if (!strcmp(argv[i],"--part"))     part = argv[++i];
        else { fprintf(stderr,"unknown arg %s\n", argv[i]); return 1; }
    }
    kern_t K = pick(N);
    if (!K){ fprintf(stderr,"n=%d not instantiated\n", N); return 1; }

    const int Lfree = 2*N - 2;
    if (chunkLog < 0) chunkLog = (Lfree < 34) ? Lfree : 34;   /* ~0.5 s per launch */
    if (chunkLog > Lfree) chunkLog = Lfree;
    while (innerBits + 8 > chunkLog) innerBits--;             /* need >=1 block */
    const int blocksLog = chunkLog - innerBits - 8;
    const int blocks = 1 << blocksLog;
    const long long totalChunks = 1LL << (Lfree - chunkLog);
    if (count < 0) count = totalChunks - from;

    build_tables(N, innerBits);
    uint32_t *d_out; CHECK(cudaMalloc(&d_out,(size_t)blocks*5*sizeof(uint32_t)));
    uint32_t *h_out = (uint32_t*)malloc((size_t)blocks*5*sizeof(uint32_t));

    u160 total; memset(&total,0,sizeof total);
    long long done = 0;

    /* resume */
    if (ckpt){
        FILE *f = fopen(ckpt,"r");
        if (f){
            int n2,cl; long long fr,ct,dn; u160 s;
            if (fscanf(f,"ckpt %d %d %lld %lld %lld %x %x %x %x %x",&n2,&cl,&fr,&ct,&dn,
                       &s.w[0],&s.w[1],&s.w[2],&s.w[3],&s.w[4])==10
                && n2==N && cl==chunkLog && fr==from && ct==count){
                total = s; done = dn;
                fprintf(stderr,"resuming from checkpoint: %lld/%lld chunks already done\n", done, count);
            } else fprintf(stderr,"checkpoint present but does not match this job; ignoring\n");
            fclose(f);
        }
    }

    fprintf(stderr,"n=%d  2n=%d  free bits=%d  chunk=2^%d sums  blocks=%d inner=2^%d\n"
                   "total chunks=%lld  running %lld..%lld\n",
            N, 2*N, Lfree, chunkLog, blocks, innerBits, totalChunks, from+done, from+count-1);

    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    double secs0 = 0;
    for (long long c = from + done; c < from + count; c++){
        uint64_t base = (uint64_t)c << chunkLog;
        K<<<blocks,256>>>(base, innerBits, d_out);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_out, d_out, (size_t)blocks*5*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int k = 0; k < blocks; k++) h_add(&total,(u160*)(h_out+5*k));
        done++;
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double el = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
        if (el - secs0 > 10.0 || done == count){
            secs0 = el;
            double rate = (double)done*(double)(1ULL<<chunkLog)/el;
            double eta  = (count-done)*el/done;
            fprintf(stderr,"  %lld/%lld chunks  %.3f Gsums/s  elapsed %.1f h  ETA %.1f h\n",
                    done, count, rate*1e-9, el/3600.0, eta/3600.0);
            if (ckpt){
                char tmp[512]; snprintf(tmp,sizeof tmp,"%s.tmp",ckpt);
                FILE *f = fopen(tmp,"w");
                if (f){ fprintf(f,"ckpt %d %d %lld %lld %lld %08x %08x %08x %08x %08x\n",
                                N,chunkLog,from,count,done,
                                total.w[0],total.w[1],total.w[2],total.w[3],total.w[4]);
                        fclose(f); rename(tmp,ckpt); }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double sums = (double)count*(double)(1ULL<<chunkLog);
    fprintf(stderr,"enumerated %.6g sign vectors in %.1f s  ->  %.4f Gsums/s\n",
            sums, el, sums/el*1e-9);

    if (part) write_part(part, N, chunkLog, from, count, total, el);

    if (from == 0 && count == totalChunks) return finish(N, total);
    printf("PARTIAL n=%d chunklog=%d chunks %lld..%lld  sum = ", N, chunkLog, from, from+count-1);
    print_i160(total); printf("\n");
    return 0;
}
