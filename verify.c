/* verify.c -- validation INDEPENDANTE de la chaine algebrique.
 *
 * Trois comptages du meme nombre, par trois chemins qui ne partagent rien :
 *
 *   1. brute(n)    -- retour sur trace, place les valeurs n..1 dans 2n cases.
 *                     Aucune algebre : c'est la definition du probleme.
 *   2. godfrey(n)  -- V(n) = 2^{-2n} sum_{X in {+-1}^{2n}} (prod x_k) F(n,X),
 *                     evalue sur TOUS les 2^{2n} points, sans aucune symetrie,
 *                     en arithmetique exacte.  Teste l'identite de Godfrey.
 *   3. klein(n)    -- la meme somme, mais en epinglant x_{2n-1}=x_{2n}=+1 et en
 *                     multipliant par 4, comme le fait le noyau GPU.  Teste la
 *                     REDUCTION par symetrie -- et montre pour quels n elle est
 *                     legitime.
 *
 * Si les trois coincident, l'identite de Godfrey ET la reduction de symetrie
 * sont verifiees a ce n.  Si 3 differe de 1 et 2, la reduction est invalide.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ---------- 1. force brute : la definition, rien d'autre ---------- */
static int NB; static long long cnt;
static void place(int v, uint64_t used){
    if (v == 0) { cnt++; return; }
    int span = v + 1;                       /* v+1 cases entre les deux copies */
    for (int p = 0; p + span < 2*NB; p++) {
        uint64_t m = (1ULL<<p) | (1ULL<<(p+span));
        if (used & m) continue;
        place(v-1, used | m);
    }
}
static long long brute(int n){
    if (2*n > 40) return -1;                /* au-dela, trop lent de toute facon */
    NB = n; cnt = 0; place(n, 0ULL); return cnt;
}

/* ---------- arithmetique exacte 128 bits ---------- */
static void print128(__int128 v){
    if (v < 0) { putchar('-'); v = -v; }
    char s[48]; int k = 0;
    if (!v) { putchar('0'); return; }
    while (v) { s[k++] = '0' + (int)(v % 10); v /= 10; }
    while (k--) putchar(s[k]);
}

static void s128(char *out, size_t cap, __int128 v){
    char t[48]; int k = 0, neg = v < 0; if (neg) v = -v;
    if (!v) t[k++] = '0';
    while (v) { t[k++] = '0' + (int)(v % 10); v /= 10; }
    if (neg) t[k++] = '-';
    int m = 0; while (k && (size_t)m + 1 < cap) out[m++] = t[--k];
    out[m] = 0;
}

/* A_i(X) pour X code par un masque b (bit a 1 = valeur -1) */
static inline int Ai(uint64_t b, int i, int M){
    int w = M - i;
    uint64_t mk = (w >= 64) ? ~0ULL : ((1ULL<<w) - 1);
    return w - 2*__builtin_popcountll((b ^ (b>>i)) & mk);
}

/* somme de Godfrey sur un sous-ensemble de points ; `fix` = nombre de bits de
 * poids fort epingles a 0 (0 = tous les points, 2 = la reduction de Klein). */
static __int128 gsum(int n, int fix){
    const int M = 2*n;
    const uint64_t NP = 1ULL << (M - fix);
    __int128 acc = 0;
    for (uint64_t b = 0; b < NP; b++) {
        __int128 prod = 1;
        for (int i = 2; i <= n+1; i++) {
            int a = Ai(b, i, M);
            if (!a) { prod = 0; break; }
            prod *= a;
        }
        if (!prod) continue;
        if (__builtin_popcountll(b) & 1) acc -= prod; else acc += prod;
    }
    return acc;
}

int main(int argc, char **argv){
    int nmax = argc > 1 ? atoi(argv[1]) : 12;
    printf("  n | mod4 | force brute V(n) |  Godfrey complet | Klein x4 (noyau) | verdict\n");
    printf("----+------+------------------+------------------+------------------+--------------------------\n");
    int bad = 0;
    for (int n = 1; n <= nmax; n++) {
        const int M = 2*n;
        long long B = brute(n);              /* compte deja les miroirs : c'est V(n) */
        __int128 Vfull = gsum(n, 0) >> M;    /* division exacte par 2^{2n} */
        __int128 Vkl   = (gsum(n, 2) * 4) >> M;
        int okF = (B < 0) || (Vfull == (__int128)B);
        int okK = (Vkl == Vfull);
        int doit = (n % 4 == 0 || n % 4 == 3); /* n ou une suite existe */
        char b1[48], b2[48], b3[48];
        if (B >= 0) snprintf(b1, sizeof b1, "%lld", B); else snprintf(b1, sizeof b1, "(hors portee)");
        s128(b2, sizeof b2, Vfull);
        s128(b3, sizeof b3, Vkl);
        printf("%3d |  %d   | %16s | %16s | %16s | ", n, n % 4, b1, b2, b3);
        if (!okF)              { printf("*** GODFREY FAUX ***");            bad = 1; }
        else if (okK &&  doit) printf("OK");
        else if (!okK && !doit) printf("Klein x4 invalide -- ATTENDU");
        else if (!okK &&  doit) { printf("*** Klein x4 invalide ICI ***");  bad = 1; }
        else                    printf("OK (Klein fortuitement juste)");
        printf("\n"); fflush(stdout);
    }
    printf("\n%s\n", bad ? "*** au moins une divergence ***"
                         : "toutes les lignes concordent");
    return bad;
}
