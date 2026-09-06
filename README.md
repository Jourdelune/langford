# Langford L(2,31)

**[Français](#français) · [English](#english)**

```
L(2,31) = 5 894 683 902 597 484 486 903 808
V(31)   = 2·L(2,31) = 11 789 367 805 194 968 973 807 616
```

Calculé le 6 septembre 2026 en 15,7 heures sur 33 GPU, pour ~50 $.
Aucune valeur de L(2,31) n'avait été publiée auparavant.

> **Ce dépôt est entièrement l'œuvre d'une IA.** Les 40 commits, du premier au
> dernier, ont été écrits par **Claude Opus 5** — algorithme, noyau CUDA,
> orchestration, vérification et cette page. Détail au §[Travail réalisé par
> IA](#travail-réalisé-par-ia).

Le journal de bord complet — quatorze pistes fermées, les mesures ratées, les
preuves de non-existence — est dans **[TRACE.md](TRACE.md)** (2 300 lignes).
Cette page en est le résumé.

---

# Français

## Le résultat

Une **suite de Langford** L(2,n) range les entiers 1…n deux fois chacun sur 2n
cases, de sorte que les deux copies de *k* soient séparées par exactement *k*
cases. Pour n=3 : `3 1 2 1 3 2`. De telles suites n'existent que si
n ≡ 0 ou 3 (mod 4) — c'est la condition de Davies (1959).

Compter *toutes* les suites pour un n donné est le problème dur. La suite des
valeurs connues est [OEIS A014552](https://oeis.org/A014552) ; elle s'arrêtait
à n=28, calculé en 2015. **n=31 était le record ouvert.**

| n | L(2,n) | qui |
|---|---|---|
| 27 | 111 683 611 098 764 903 232 | Assarpour, Bar-Noy & Liu (2015) |
| 28 | 1 607 383 260 609 382 393 152 | Assarpour, Bar-Noy & Liu (2015) |
| **31** | **5 894 683 902 597 484 486 903 808** | **ce dépôt (2026)** |

Le calcul a été découpé en 8 192 tranches plus une tâche « diagonale »,
réparties sur 32 RTX 4070 Super louées et une RTX 4070 locale.

## Vérifier le résultat soi-même

Un nombre sans moyen de le contredire ne vaut rien. Tout ce qui suit est fourni
pour qu'un tiers puisse recalculer, pas seulement relire.

### Le dossier de preuve

`preuve_n31_20260906T053939Z.tar.gz` — 944 Ko

```
sha256  ef66b1a3d699c7086634f31e02ada0636fcee7ea3a8039e7a59135e648e5806a
```

| fichier | sha256 (tronqué) | contenu |
|---|---|---|
| `parts_n31.txt` | `1765a9c05338d642…` | les 8 193 sommes partielles, **avec la provenance de chacune** : carte, pilote, empreinte du binaire, empreinte de la source, durée, horodatage UTC |
| `resultat.txt` | `a03a6eba950182bd…` | la sortie du merge, avec son auto-test |
| `state.db` | `434acc15be8f80da…` | instantané SQLite de l'état complet du run |
| `audit.txt` | `f53143ae46496e24…` | le dossier d'audit intégral |
| `manifeste.json` | `7762717c17865e78…` | commit git, sha256 de la source, inventaire des workers et des machines louées |

Chaque fichier est haché dans `SHA256SUMS` ; `sha256sum -c SHA256SUMS` vérifie
le tout.

### Les quatre contrôles, et ce que chacun attrape

**1. Complétude et unicité.** Les 8 192 tranches et la diagonale sont présentes,
aucune en double. *Attrape* : une tranche perdue ou comptée deux fois.

**2. Auto-test arithmétique.** Chaque terme de la somme est un produit des n
facteurs A_i, et A_i ≡ i (mod 2) ; les ⌊(n+1)/2⌋ écarts pairs donnent chacun un
facteur 2. Le total doit donc être divisible par 2^(2n+1) = 2^63. Une seule
tranche corrompue fait échouer le test. *Attrape* : une somme partielle altérée
en transit ou en mémoire. Exécuté par `langford6 --merge` **avant** toute
destruction de machine.

**3. Recalcul redondant.** Douze tranches tirées au hasard ont été **refaites
localement** sur une autre carte et un autre binaire, puis comparées bit à bit
à ce qu'avaient rendu les machines louées :

```
tache 1668 3242 7817 4421 4142 1861 250 3783 2673 4896 3386 5204   IDENTIQUE
-> l'echantillon se reproduit au bit pres          (12/12)
```

*Attrape* : une carte défaillante, un binaire différent, une tranche calculée
sur le mauvais intervalle.

**4. Croisement de deux algorithmes** — le plus fort. `refvals.txt` contient des
tranches produites **deux fois par deux algorithmes indépendants** : le noyau
GPU (identité de Godfrey) et `slice_ref` (backtracking classique). Sur n=31 aux
trois régimes extrêmes de `vhi`, plus n=28, 27, 24, 20 :

```
$ ./check_refs.sh
  n=31  vhi=0         OK        n=28  vhi=1048575  OK
  n=31  vhi=4194304   OK        n=27  vhi=524287   OK
  n=31  vhi=8388607   OK        n=24  vhi=65535    OK
                                n=20  vhi=63       OK
  7 valeur(s) de reference confirmee(s), 0 divergence(s)
```

*Attrape* : une erreur dans la méthode elle-même, pas seulement dans son
exécution. C'est le seul contrôle qui ne suppose pas que Godfrey est bien
implémenté.

### Le repère indépendant

`estimate.c` implémente l'**estimateur non biaisé de Knuth (1975)** pour la
taille d'un arbre de backtracking : on tire un chemin racine→feuille au hasard,
en multipliant le poids par le nombre d'enfants légaux à chaque nœud.
L'espérance du poids vaut exactement le nombre de feuilles.

| tirages | estimation de L(2,31) | erreur-type (1σ) | écart au calcul |
|---|---|---|---|
| 2 · 10⁶ | 5,74000 · 10²⁴ | ± 2,00 % | +2,695 % (1,35 σ) |
| 2 · 10⁷ | 5,87786 · 10²⁴ | ± 0,38 % | +0,286 % (0,75 σ) |
| 2 · 10⁸ | 5,88528 · 10²⁴ | ± 0,12 % | **+0,160 %** (1,33 σ) |

L'écart relatif fond de façon monotone — 2,7 % puis 0,29 % puis 0,16 % — et
l'estimation monte à chaque fois **vers** la valeur calculée. C'est le
comportement attendu : la distribution des poids est très asymétrique à droite,
la médiane d'un tirage fini tombe donc sous la vraie valeur, et un petit
échantillon sous-estime systématiquement. Un Monte-Carlo indépendant qui tombe à
0,16 % d'un entier de 25 chiffres est un bon repère — mais **un repère
seulement** : il ne peut jamais confirmer les derniers chiffres.

### Refaire le calcul

```sh
./build.sh                                    # compile tout
./langford6 -n 12                             # doit afficher 108144
./verify_all.sh                               # la chaîne complète sur les petits n
./check_refs.sh                               # croisement des deux algorithmes
./audit.sh 31 8192 parts_n31.txt              # le dossier d'audit complet

# le calcul lui-même, une tranche à la fois (rejouable sans état)
./run_shard.sh <i> 8192 31                    # i de 0 à 8191
./langford6 -n 31 --diag                      # la tâche diagonale
./langford6 -n 31 --merge-file parts.txt      # conclure
```

## La méthode, expliquée

### 1. Compter devient sommer — l'identité de Godfrey (2002)

Une suite de Langford est un couplage parfait des 2n positions dont le
multi-ensemble des écarts vaut exactement {2, 3, …, n+1}. Godfrey encode ça dans
un polynôme :

```
F(n,X) = ∏_{i=2}^{n+1} A_i(X)      avec   A_i(X) = Σ_{k=1}^{2n-i} x_k · x_{k+i}
```

`A_i` énumère toutes les façons de placer une paire d'écart *i*. Le nombre de
suites est le coefficient du monôme x₁x₂…x_{2n} dans F.

L'astuce : F est homogène de degré 2n en 2n variables, donc **tout monôme qui
n'est pas x₁x₂…x_{2n} possède une variable d'exposant pair**. En substituant
x_k = ±1 et en sommant sur les 2^{2n} assignations, tous ces monômes s'annulent
et il ne reste que celui qu'on cherche :

```
V(n) = 2·L(2,n) = 2^{-2n} · Σ_{X ∈ {-1,1}^{2n}} (∏_k x_k) · F(n,X)
```

Un problème de comptage combinatoire devient **une somme de 4ⁿ termes entiers**.
Chaque terme est indépendant : c'est parfait pour un GPU. Coût Θ(4ⁿ), inchangé
depuis 2002 — et le §5 de [TRACE.md](TRACE.md) explique, en quatorze voies
fermées, pourquoi personne ne sait faire mieux.

### 2. Le groupe de symétries d'ordre 8

Deux symétries laissent la somme invariante :

- **le groupe de Klein** — nier toutes les variables de rang pair, ou de rang
  impair, ne change pas le produit ;
- **la réflexion** p ↦ 2n+1−p, qui renverse la suite.

Ensemble : un groupe d'ordre 8. On ne calcule donc qu'un huitième des points, et
on multiplie par 8. L'article de 2015 liste ces symétries mais n'en tire qu'un
facteur 4 ; le facteur 8 complet vaut **×2,00** de plus.

Attention : cette réduction n'est **valide que si n ≡ 0 ou 3 (mod 4)**. Nier une
rangée multiplie le produit par (−1)^MO et le poids par (−1)^N ; la sommande
n'est invariante que si MO + N est pair. Aux autres n le noyau rendrait un
nombre parfaitement faux et parfaitement crédible — pour n=9 il annonçait 5558
là où la réponse est 0. Le programme **refuse** donc ces n.

### 3. La décomposition de parité — l'idée structurelle

Séparons les positions **impaires** (rangée `o`, n cases) des **paires**
(rangée `e`). Un calcul d'indices donne, pour tout n :

```
écart PAIR   i = 2m   :  A_i = P_m(o) + Q_m(e)                    ← rangées SÉPARÉES
écart IMPAIR i = 2m+1 :  A_i = Σ_j O_j·E_{j+m} + Σ_j E_j·O_{j+m+1}  ← bilinéaire
```

où P_m et Q_m sont les autocorrélations de rangée au lag m. Vérifié exactement
sur 6,2 millions d'écarts jusqu'à n=31, zéro divergence.

Deuxième observation, décisive : `A_i` est une somme de (2n−i) termes ±1, donc
**A_i ≡ i (mod 2)**. Seuls les écarts **pairs** peuvent s'annuler — et c'est
l'annulation d'un facteur qui décide qu'un point ne contribue pas.

Mises bout à bout : **la survie d'un point ne dépend de la rangée `o` que par un
vecteur constant.** Pour un `e_hi` fixé, la table `e_lo → Q(e)` ne dépend pas du
tout de `o` et se calcule une fois pour tout un bloc de threads. La boucle chaude
— la mise à jour incrémentale d'état à chaque point — **disparaît**, remplacée
par des *bitmaps de survie*. C'est la seule idée vraiment structurelle du dépôt :
**×2,21**.

La décomposition deux rangées était connue (elle donne l'argument classique
n ≡ 0,3 mod 4) mais tenue pour sans gain algorithmique. L'exploiter comme
*séparation de variables* est ce qui est neuf.

### 4. Les 87 % de produits nuls

Un point sur huit seulement contribue : les autres ont au moins un facteur nul.
L'observation est ancienne, mais elle était rejetée comme inexploitable en SIMT —
dans un warp, si un seul thread sur 32 a du travail, les 31 autres attendent
quand même.

La **compaction par warp** (technique NVIDIA classique) lève l'objection : les
survivants sont rassemblés en file avant d'être traités, si bien que le travail
utile est dense. **×1,77**.

## Les optimisations, une par une

Toutes les lignes donnent le coût de **n=31 sur la même RTX 4070**, ce qui rend
les gains comparables. Chaque facteur est le gain sur la ligne précédente.

| # | ce qui change | n=31 | gain | origine |
|---|---|---|---|---|
| | **Godfrey nu (2002)** — pas de symétrie, modulaire + CRT | ≈ 4 500 j | — | littérature |
| 1 | **symétrie d'ordre 4** → l'état de l'art publié (2015) | ≈ 1 100 j | ×4,1 | littérature (Assarpour *et al.* §4) |
| 2 | **bignum 160 bits tronqué** au lieu de modulaire + CRT | 360 j | ×3,06 | hors Langford (Knuth TAOCP II §4.3.1) |
| 3 | **symétrie d'ordre 8** — la réflexion en plus | 180,1 j | ×2,00 | mixte |
| 4 | **saut des produits nuls** + compaction par warp | 101,7 j | ×1,77 | mixte |
| 5 | **demi-état** — seuls les écarts pairs décident | 75,0 j | ×1,36 | dérivé de A_i ≡ i (mod 2) |
| 6 | **parité + bitmaps de survie** — la boucle chaude disparaît | 34,0 j | ×2,21 | **neuf ici** |
| 7 | **chaînes de retenue PTX** + extraction `prmt` | 30,9 j | ×1,10 | hors Langford |
| 8 | **écarts impairs tabulés** sur l'axe `threadIdx` | **25,8 j** | ×1,20 | **neuf ici** |

> **De l'état de l'art publié (2015) à ce dépôt, à matériel identique : ×42,6.**
> Depuis Godfrey nu : ×174. Mesure directe de bout en bout : ×45.

Une seule ligne est un **modèle** et non une mesure : la ligne 2. La version
publiée n'est pas réimplémentée ici, son coût est reconstitué à partir de ses
deux écarts avec la v3. L'incertitude qui en découle donne une fourchette
honnête de **×29 à ×54** pour le total. Les lignes 1 et 3 sont exactes *par
construction* (elles ne font que compter les points énumérés) ; les lignes 4 à 8
sont des rapports mesurés, GPU au repos, appariés sur le même échantillon.

**L'essentiel du gain est de l'ingénierie, pas une idée.** Une seule ligne est
structurelle (×2,21) ; tout le reste — arithmétique, symétrie complète,
compaction, PTX, tabulation — vaut ×19,4 à lui seul.

### Deux résultats négatifs, chiffrés

Ils ont coûté du temps et méritent d'être publiés, ne serait-ce que pour éviter
à quelqu'un de les refaire :

- **Kasteleyn ne s'applique pas.** Une orientation pfaffienne donnerait 2ⁿ·n³ au
  lieu de 4ⁿ et le record tomberait en minutes. Réfuté sous la forme faible et
  correcte du problème, sur GF(2) **et** sur Z/2^k jusqu'à k=32.
- **Le tensor core ne bat pas `popc`.** Le réflexe « c'est bilinéaire donc c'est
  un GEMM » est faux ici : `popc` sur un XOR 32 bits *est* un produit scalaire
  binaire de longueur 32, l'INT8 ne mène que ×2,48 sur cette carte, et la
  sparsité (12,9 % de survivants, qu'un MMA dense ne sait pas exploiter)
  retourne l'avantage en **×4,1 de retard**.

## Comparaison avec l'état de l'art 2015

Sur **L(28)**, le dernier point du record publié — donc à instance identique :

| | matériel | temps | GPU-jours |
|---|---|---|---|
| Assarpour, Bar-Noy & Liu (2015) | ~32 GPU Kepler | ~9 jours | ~288 |
| **ce dépôt** | 1 × RTX 4070 | **≈ 10,6 h** | **0,44** |

soit **~650× moins de GPU-jours**. Il faut décomposer honnêtement :

- **6 à 11×** viennent du **matériel** — un Kepler de 2013 délivre ~1,3·10¹²
  opérations entières/s contre ~7,3·10¹² pour une 4070 ;
- **~45×** viennent de l'**algorithme et de l'implémentation**, chiffrés à
  matériel identique ;
- le reste absorbe le rendement de la distribution sur ~32 cartes.

**Asymptotiquement, il n'y a aucun progrès.** L'algorithme reste Θ(4ⁿ) et
l'exposant n'a pas bougé depuis 2002. Ce dépôt gagne des **constantes**, pas un
exposant, et ne prétend pas le contraire.

## Et n=32 ?

n=32 ≡ 0 (mod 4), donc des suites existent. Le coût est en 4ⁿ : **exactement 4×
n=31**. L'estimateur donne, sur 2·10⁷ tirages :

```
L(2,32) ≈ 9,744 · 10²⁵   ± 0,45 %        log2(V) = 87,333
```

### Le temps GPU nécessaire

C'est la seule quantité qui ne dépende pas du nombre de cartes qu'on aligne.
Elle se déduit sans incertitude : le coût est en 4ⁿ, donc **n=32 = 4 × n=31**,
et n=31 a été mesuré sur quatre cartes par le même bench à graine fixe (même
échantillon de `vhi` des deux côtés, donc comparaison appariée).

| carte | n=31 | **n=32 = ×4** | provenance du chiffre |
|---|---|---|---|
| RTX 4070 | 544 h·GPU | **2 177 h·GPU** | mesuré, source actuelle |
| RTX 4070 Super | 497 h·GPU | **1 988 h·GPU** | mesuré, source actuelle ; **confirmé à +0,2 % par le run réel** |
| RTX 4090 | ~204 h·GPU | ~815 h·GPU | déduit du rapport apparié ×2,67 |
| RTX 5090 | ~174 h·GPU | ~695 h·GPU | déduit du rapport apparié ×3,13 |

**Environ 2 000 heures·GPU sur une 4070 Super ; ~800 sur une 4090.** À titre de
comparaison, n=31 en a consommé 498 sur les cartes louées, contre 497 annoncées
avant location — c'est cette concordance à 0,2 % qui donne confiance dans la
projection ×4.

Les deux premières lignes sont mesurées avec le binaire qui a produit le
résultat, sur le même échantillon de 96 `vhi` à graine fixe. Les deux dernières
ne le sont pas : leurs absolus datent d'un build antérieur à la v7, et seul le
**rapport** entre cartes est réputé stable (§7.1 de [TRACE.md](TRACE.md)). Elles
sont donc obtenues en appliquant ce rapport à la mesure 4070 actuelle, et
devraient être revérifiées avant d'engager une location.

Le temps de paroi n'est qu'une division : la même flotte de 32 × 4070 Super le
ferait en **62 h**, 128 cartes en 16 h. Le coût en dollars ne dépend pas non plus
de la flotte — **≈ 190 $** à 0,0945 $/GPU-h, le tarif effectivement payé ici.
n=32 est donc **à portée immédiate**.

### Ce qu'il faudrait comme code — trois obstacles réels

**1. Le noyau n'est pas instancié pour 32.** `pick()` et `dpick()` ne couvrent
que {9…24, 27, 28, 31}. Il faut ajouter `INST(32)` et `DINST(32)`.

**2. Un dépassement silencieux dans le masque de rangée.** Les rangées sont des
`uint32_t` et le code calcule :

```c
const uint32_t ALLB0 = (1u<<N)-1u;      // N=32  ->  1u<<32
```

Un décalage de 32 sur un type de 32 bits est un **comportement indéfini** en
C/C++. Sur le matériel courant il rend 1 (décalage modulo 32), donc `ALLB0`
vaudrait 0 et la négation de rangée serait silencieusement fausse. Il faut soit
un cas particulier, soit élargir la représentation. **n=32 est le dernier n qui
tient dans un `uint32_t`** : 32 cases par rangée, 31 bits libres après épinglage.

**3. La marge de l'accumulateur se resserre.** Le total vaut V·2^{2n}, donc :

| n | log2(V) | + 2n | bits utilisés / 160 | marge |
|---|---|---|---|---|
| 31 | 83,3 | 62 | 145,3 | 14,7 bits |
| 32 | 87,3 | 64 | **151,3** | **8,7 bits** |

Ça passe, mais il ne reste que ~8,7 bits. Pour n=35 (le suivant ≡ 3 mod 4),
l'accumulateur 160 bits serait insuffisant.

Le vrai mur reste le 4ⁿ : chaque n de plus coûte 4×, et n=35 coûterait 256× n=31,
soit ~127 000 h·GPU et ~12 000 $. Le §5 de [TRACE.md](TRACE.md) argumente que
casser cet exposant demanderait un mécanisme inconnu.

## Coût de l'expérience

| poste | montant |
|---|---|
| calcul — 15,7 h × 3,044 $/h (4 machines × 8 RTX 4070 Super) | 47,74 $ |
| locations avortées (une 14×4090 préemptée, une 8×4090 frauduleuse, reprises de tunnel) | ≈ 2 $ |
| RTX 4070 locale | électricité |
| **total** | **≈ 50 $** |

Le calcul a consommé **498 h·GPU** sur les cartes louées, contre **497 h**
annoncées par le bench avant location : **+0,2 %** d'écart. La répartition entre
les quatre machines est restée équilibrée (2 027 / 1 998 / 1 970 / 1 957 tâches),
et la 4070 locale en a absorbé 238.

Une observation en passant : les 4070 Super louées ne se sont montrées que
**9,5 % plus rapides** qu'une 4070 (497 h contre 544), là où leur nombre de SM
— 56 contre 46 — en laissait attendre 22 %. Elles tiraient ~160 W pour un TDP de
220 W : des cartes bridées en puissance. Le prix par carte restait imbattable, la
conclusion ne change pas, mais un `$/GPU-h` ne dit rien du débit réel.

Une leçon a été payée comptant : l'offre la moins chère du marché, une « 8×
RTX 4090 » à 1,72 $/h, **falsifiait `nvidia-smi`** via un `/opt/fake/nvml_reader.py`
et n'exécutait aucun noyau sm_89. Le rabais était le symptôme. L'orchestrateur
refuse désormais ces hôtes automatiquement et n'accepte une machine qu'après lui
avoir fait calculer L(2,12) = 108144.

## Travail réalisé par IA

**L'intégralité de ce dépôt a été produite par Claude Opus 5**, d'Anthropic,
depuis le premier commit. Les 40 commits portent tous
`Co-Authored-By: Claude Opus 5`, sur trois sessions entre le 4 et le 6 septembre
2026.

Cela couvre : la dérivation de la décomposition de parité et sa preuve, le noyau
CUDA et son assembleur PTX, les quatorze pistes de recherche explorées puis
fermées (dont les deux résultats négatifs ci-dessus), l'orchestrateur distribué
et sa tolérance aux préemptions, la chaîne de vérification, la location et le
pilotage des GPU chez un fournisseur externe, et cette page.

Le rôle humain a été de fixer les objectifs, de fournir l'accès au matériel et
au compte de location, et de trancher les arbitrages de dépense.

Ce qui n'a **pas** été délégué à l'IA : la vérification par des tiers. C'est
précisément pourquoi le §[Vérifier le résultat soi-même](#vérifier-le-résultat-soi-même)
existe et pourquoi chaque somme partielle porte sa provenance. Un résultat
produit par une IA n'a pas moins besoin d'être recalculé par quelqu'un
d'autre — il en a davantage besoin.

## Références

**Le problème**

- C. D. Langford, *Problem*, Math. Gazette **42** (1958), 228.
- R. O. Davies, *On Langford's problem II*, Math. Gazette **43** (1959), 253–255. — condition d'existence n ≡ 0, 3 (mod 4)
- T. Skolem, *On certain distributions of integers in pairs with given differences*, Math. Scand. **5** (1957), 57–68.
- [OEIS A014552](https://oeis.org/A014552) — L(2,n) ; [A059106](https://oeis.org/A059106) — variante de Skolem.

**La méthode de comptage**

- M. Godfrey, méthode algébrique (2002). Pas de publication formelle ; décrite dans Assarpour–Bar-Noy–Liu §3 et dans D. E. Knuth, *TAOCP* vol. 4, pré-fascicule 5B, section « Langford pairs ».
- A. Assarpour, A. Bar-Noy, O. Liu, *Counting Skolem Sequences*, [arXiv:1507.00315](https://arxiv.org/abs/1507.00315) (2015, rév. 2017). — L(27) et L(28), implémentation CUDA de Godfrey ; **c'est l'état de l'art auquel ce dépôt se compare**
- M. Krajecki, C. Jaillet, A. Bui *et al.*, calculs distribués CONFIIT (2004–2005) — valeurs jusqu'à n=24.
- D. E. Knuth, *Estimating the efficiency of backtrack programs*, Math. Comp. **29** (1975), 121–136. — l'estimateur non biaisé de `estimate.c`

**Les pistes évaluées puis fermées**

- P. W. Kasteleyn, *The statistics of dimers on a lattice*, Physica **27** (1961) ; *Dimer statistics and phase transitions*, J. Math. Phys. **4** (1963). — orientations pfaffiennes
- L. Lovász, M. D. Plummer, *Matching Theory*, ch. 8.
- A. Björklund, *Counting Perfect Matchings as Fast as Ryser*, SODA 2012, [arXiv:1107.4466](https://arxiv.org/abs/1107.4466).
- *Counting perfect matchings and Hamiltonian cycles faster*, [arXiv:2309.15422](https://arxiv.org/abs/2309.15422) (2023).
- *A New Direction for Counting Perfect Matchings*, [arXiv:1208.2329](https://arxiv.org/abs/1208.2329).
- M. Cygan *et al.*, *On problems as hard as CNF-SAT* / Set Cover Conjecture (2016) — la barrière 2^{|U|}.

**Techniques d'implémentation**

- F. Gray, brevet US 2632058 (1953) — code binaire réfléchi.
- R. J. Fisher, H. G. Dietz, *Compiling for SIMD Within A Register*, LCPC 1998.
- S. E. Anderson, *Bit Twiddling Hacks*.
- D. E. Knuth, *TAOCP* vol. 2, §4.3.1 — multiplication multi-précision.
- NVIDIA, *CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics* — compaction de flux par warp.

## Fichiers

| fichier | rôle |
|---|---|
| `langford6.cu` | le noyau CUDA, toutes optimisations |
| `slice_ref.c` | l'implémentation de référence, algorithme *différent* |
| `orchestrator.py` | location, distribution par baux, reprise après préemption |
| `finalize.py` | clôture : mise à l'abri, libération des machines, preuve |
| `dashboard.py` | avancement sur `http://localhost:8800` |
| `audit.sh` | le dossier d'audit, recalcul redondant compris |
| `check_refs.sh` | croisement des deux algorithmes |
| `verify_all.sh` | la chaîne complète sur les petits n |
| `estimate.c` | estimateur de Knuth, repère indépendant |
| **[TRACE.md](TRACE.md)** | **le journal de bord complet, 2 300 lignes** |

---

# English

## The result

A **Langford pairing** L(2,n) arranges the integers 1…n, each twice, over 2n
slots so that the two copies of *k* are separated by exactly *k* slots. For n=3:
`3 1 2 1 3 2`. Such sequences exist only when n ≡ 0 or 3 (mod 4) — the Davies
condition (1959).

Counting *all* pairings for a given n is the hard problem.
[OEIS A014552](https://oeis.org/A014552) lists the known values; it stopped at
n=28, computed in 2015. **n=31 was the open record.**

| n | L(2,n) | by |
|---|---|---|
| 27 | 111 683 611 098 764 903 232 | Assarpour, Bar-Noy & Liu (2015) |
| 28 | 1 607 383 260 609 382 393 152 | Assarpour, Bar-Noy & Liu (2015) |
| **31** | **5 894 683 902 597 484 486 903 808** | **this repository (2026)** |

The computation was split into 8,192 slices plus one "diagonal" task, spread
across 32 rented RTX 4070 Supers and one local RTX 4070.

## Verifying the result yourself

A number with no way to contradict it is worthless. Everything below is provided
so a third party can *recompute*, not merely read.

### The proof bundle

`preuve_n31_20260906T053939Z.tar.gz` — 944 KB

```
sha256  ef66b1a3d699c7086634f31e02ada0636fcee7ea3a8039e7a59135e648e5806a
```

| file | sha256 (truncated) | contents |
|---|---|---|
| `parts_n31.txt` | `1765a9c05338d642…` | all 8,193 partial sums, **each with its provenance**: card, driver, binary hash, source hash, duration, UTC timestamp |
| `resultat.txt` | `a03a6eba950182bd…` | merge output with its self-test |
| `state.db` | `434acc15be8f80da…` | SQLite snapshot of the full run state |
| `audit.txt` | `f53143ae46496e24…` | the complete audit dossier |
| `manifeste.json` | `7762717c17865e78…` | git commit, source sha256, inventory of workers and rented machines |

Every file is hashed in `SHA256SUMS`; `sha256sum -c SHA256SUMS` checks the lot.

### The four checks, and what each one catches

**1. Completeness and uniqueness.** All 8,192 slices plus the diagonal are
present, none duplicated. *Catches*: a lost or double-counted slice.

**2. Arithmetic self-test.** Each summand is a product of the n factors A_i, and
A_i ≡ i (mod 2); the ⌊(n+1)/2⌋ even gaps each contribute a factor of 2. The
total must therefore be divisible by 2^(2n+1) = 2^63. One corrupted slice makes
the test fail. *Catches*: a partial sum altered in transit or in memory. Run by
`langford6 --merge` **before** any machine is destroyed.

**3. Redundant recomputation.** Twelve randomly drawn slices were **recomputed
locally** on a different card with a different binary, then compared bit-for-bit
against what the rented machines returned:

```
task 1668 3242 7817 4421 4142 1861 250 3783 2673 4896 3386 5204   IDENTICAL
-> the sample reproduces bit-for-bit          (12/12)
```

*Catches*: a faulty card, a different binary, a slice computed over the wrong
interval.

**4. Cross-checking two algorithms** — the strongest. `refvals.txt` holds slices
produced **twice by two independent algorithms**: the GPU kernel (Godfrey's
identity) and `slice_ref` (classical backtracking). For n=31 at the three
extreme `vhi` regimes, plus n=28, 27, 24, 20:

```
$ ./check_refs.sh
  n=31  vhi=0         OK        n=28  vhi=1048575  OK
  n=31  vhi=4194304   OK        n=27  vhi=524287   OK
  n=31  vhi=8388607   OK        n=24  vhi=65535    OK
                                n=20  vhi=63       OK
  7 reference value(s) confirmed, 0 divergence(s)
```

*Catches*: an error in the method itself, not just in its execution. This is the
only check that does not assume Godfrey's identity is correctly implemented.

### The independent yardstick

`estimate.c` implements **Knuth's unbiased estimator (1975)** for backtracking
tree size: walk a random root-to-leaf path, multiplying the running weight by the
number of legal children at each node. The expected weight is exactly the number
of leaves.

| trials | estimate of L(2,31) | std. error (1σ) | gap to computed value |
|---|---|---|---|
| 2 · 10⁶ | 5.74000 · 10²⁴ | ± 2.00 % | +2.695 % (1.35 σ) |
| 2 · 10⁷ | 5.87786 · 10²⁴ | ± 0.38 % | +0.286 % (0.75 σ) |
| 2 · 10⁸ | 5.88528 · 10²⁴ | ± 0.12 % | **+0.160 %** (1.33 σ) |

The relative gap shrinks monotonically — 2.7 %, then 0.29 %, then 0.16 % — and
the estimate rises **toward** the computed value each time. That is the expected
behaviour: the weight distribution is heavily right-skewed, so the median of a
finite sample falls below the true value and a small sample systematically
underestimates. An independent Monte-Carlo landing within 0.16 % of a 25-digit
integer is a good yardstick — but **only** a yardstick: it can never confirm the
trailing digits.

### Reproducing the computation

```sh
./build.sh                                    # build everything
./langford6 -n 12                             # must print 108144
./verify_all.sh                               # full chain on small n
./check_refs.sh                               # cross-check the two algorithms
./audit.sh 31 8192 parts_n31.txt              # the complete audit dossier

# the computation itself, one slice at a time (stateless, replayable)
./run_shard.sh <i> 8192 31                    # i from 0 to 8191
./langford6 -n 31 --diag                      # the diagonal task
./langford6 -n 31 --merge-file parts.txt      # conclude
```

## The method, explained

### 1. Counting becomes summing — Godfrey's identity (2002)

A Langford pairing is a perfect matching of the 2n positions whose multiset of
gaps is exactly {2, 3, …, n+1}. Godfrey encodes this in a polynomial:

```
F(n,X) = ∏_{i=2}^{n+1} A_i(X)      where   A_i(X) = Σ_{k=1}^{2n-i} x_k · x_{k+i}
```

`A_i` enumerates every way to place a pair with gap *i*. The number of pairings
is the coefficient of the monomial x₁x₂…x_{2n} in F.

The trick: F is homogeneous of degree 2n in 2n variables, so **every monomial
other than x₁x₂…x_{2n} has some variable at an even exponent**. Substituting
x_k = ±1 and summing over all 2^{2n} assignments annihilates all of them,
leaving only the one we want:

```
V(n) = 2·L(2,n) = 2^{-2n} · Σ_{X ∈ {-1,1}^{2n}} (∏_k x_k) · F(n,X)
```

A combinatorial counting problem becomes **a sum of 4ⁿ integer terms**. Each
term is independent — ideal for a GPU. Cost Θ(4ⁿ), unchanged since 2002; §5 of
[TRACE.md](TRACE.md) explains, across fourteen closed avenues, why nobody knows
how to do better.

### 2. The order-8 symmetry group

Two symmetries leave the sum invariant:

- **the Klein group** — negating all even-indexed, or all odd-indexed, variables
  does not change the product;
- **the reflection** p ↦ 2n+1−p, which reverses the sequence.

Together: a group of order 8. So only one eighth of the points is computed, then
multiplied by 8. The 2015 paper lists these symmetries but extracts only a factor
of 4; the full factor of 8 is worth a further **×2.00**.

Caution: this reduction is **valid only when n ≡ 0 or 3 (mod 4)**. Negating a row
multiplies the product by (−1)^MO and the weight by (−1)^N; the summand is
invariant only when MO + N is even. At other n the kernel would return a
perfectly wrong and perfectly plausible number — for n=9 it reported 5558 where
the answer is 0. The program therefore **refuses** those n.

### 3. The parity decomposition — the structural idea

Separate the **odd** positions (row `o`, n cells) from the **even** ones (row
`e`). An index computation gives, for every n:

```
EVEN gap i = 2m   :  A_i = P_m(o) + Q_m(e)                     ← rows SEPARATED
ODD  gap i = 2m+1 :  A_i = Σ_j O_j·E_{j+m} + Σ_j E_j·O_{j+m+1}   ← bilinear
```

where P_m and Q_m are the row autocorrelations at lag m. Verified exactly over
6.2 million gaps up to n=31, zero divergence.

The second, decisive observation: `A_i` is a sum of (2n−i) terms of ±1, hence
**A_i ≡ i (mod 2)**. Only **even** gaps can vanish — and it is a vanishing factor
that decides a point contributes nothing.

Put together: **a point's survival depends on row `o` only through a constant
vector.** For a fixed `e_hi`, the table `e_lo → Q(e)` does not depend on `o` at
all and is computed once for a whole block of threads. The hot loop — the
incremental state update at every point — **disappears**, replaced by *survival
bitmaps*. This is the repository's one genuinely structural idea: **×2.21**.

The two-row decomposition was known (it yields the classical n ≡ 0,3 mod 4
argument) but was believed to carry no algorithmic gain. Exploiting it as a
*separation of variables* is what is new.

### 4. The 87 % null products

Only one point in eight contributes; the rest have at least one vanishing factor.
The observation is old, but it was dismissed as unexploitable under SIMT — within
a warp, if only one thread in 32 has work, the other 31 wait anyway.

**Warp compaction** (a standard NVIDIA technique) removes the objection:
survivors are gathered into a queue before processing, so the useful work is
dense. **×1.77**.

## The optimizations, one by one

Every row gives the cost of **n=31 on the same RTX 4070**, which makes the gains
comparable. Each factor is the gain over the previous row.

| # | what changes | n=31 | gain | origin |
|---|---|---|---|---|
| | **Bare Godfrey (2002)** — no symmetry, modular + CRT | ≈ 4,500 d | — | literature |
| 1 | **order-4 symmetry** → the published state of the art (2015) | ≈ 1,100 d | ×4.1 | literature (Assarpour *et al.* §4) |
| 2 | **truncated 160-bit bignum** instead of modular + CRT | 360 d | ×3.06 | outside Langford (Knuth TAOCP II §4.3.1) |
| 3 | **order-8 symmetry** — reflection as well | 180.1 d | ×2.00 | mixed |
| 4 | **skipping null products** + warp compaction | 101.7 d | ×1.77 | mixed |
| 5 | **half-state** — only even gaps decide | 75.0 d | ×1.36 | derived from A_i ≡ i (mod 2) |
| 6 | **parity + survival bitmaps** — the hot loop disappears | 34.0 d | ×2.21 | **new here** |
| 7 | **PTX carry chains** + `prmt` extraction | 30.9 d | ×1.10 | outside Langford |
| 8 | **odd gaps tabulated** on the `threadIdx` axis | **25.8 d** | ×1.20 | **new here** |

> **From the published state of the art (2015) to this repository, on identical
> hardware: ×42.6.** From bare Godfrey: ×174. Direct end-to-end measurement:
> ×45.

Exactly one row is a **model** rather than a measurement: row 2. The published
version is not reimplemented here; its cost is reconstructed from its two
differences with v3. The resulting uncertainty gives an honest range of **×29 to
×54** for the total. Rows 1 and 3 are exact *by construction* (they merely count
the enumerated points); rows 4–8 are measured ratios, GPU idle, paired on the
same sample.

**Most of the gain is engineering, not an idea.** One row is structural (×2.21);
everything else — arithmetic, full symmetry, compaction, PTX, tabulation — is
worth ×19.4 on its own.

### Two negative results, quantified

They cost time and deserve publishing, if only to spare someone else the trip:

- **Kasteleyn does not apply.** A Pfaffian orientation would give 2ⁿ·n³ instead
  of 4ⁿ and the record would fall in minutes. Refuted under the weak, correct
  form of the problem, over GF(2) **and** over Z/2^k up to k=32.
- **Tensor cores do not beat `popc`.** The reflex "it's bilinear, so it's a GEMM,
  so it's fast" is wrong here: `popc` on a 32-bit XOR *is* a length-32 binary dot
  product, INT8 leads by only ×2.48 on this card, and the sparsity (12.9 %
  survivors, which a dense MMA cannot exploit) turns that into a **×4.1
  deficit**.

## Comparison with the 2015 state of the art

On **L(28)**, the last point of the published record — i.e. on an identical
instance:

| | hardware | time | GPU-days |
|---|---|---|---|
| Assarpour, Bar-Noy & Liu (2015) | ~32 Kepler GPUs | ~9 days | ~288 |
| **this repository** | 1 × RTX 4070 | **≈ 10.6 h** | **0.44** |

That is **~650× fewer GPU-days**. It has to be decomposed honestly:

- **6–11×** comes from **hardware** — a 2013 Kepler delivers ~1.3·10¹² integer
  ops/s against ~7.3·10¹² for a 4070;
- **~45×** comes from the **algorithm and implementation**, measured on identical
  hardware;
- the remainder absorbs the efficiency of their distribution across ~32 cards.

**Asymptotically there is no progress at all.** The algorithm remains Θ(4ⁿ) and
the exponent has not moved since 2002. This repository wins **constants**, not an
exponent, and claims nothing else.

## What about n=32?

n=32 ≡ 0 (mod 4), so pairings exist. Cost goes as 4ⁿ: **exactly 4× n=31**. The
estimator gives, over 2·10⁷ trials:

```
L(2,32) ≈ 9.744 · 10²⁵   ± 0.45 %        log2(V) = 87.333
```

### The GPU time required

This is the one quantity independent of how many cards you line up. It follows
without uncertainty: cost goes as 4ⁿ, so **n=32 = 4 × n=31**, and n=31 was
measured on four cards by the same fixed-seed benchmark (identical `vhi` sample
on both sides, hence a paired comparison).

| card | n=31 | **n=32 = ×4** | where the figure comes from |
|---|---|---|---|
| RTX 4070 | 544 GPU-h | **2,177 GPU-h** | measured, current source |
| RTX 4070 Super | 497 GPU-h | **1,988 GPU-h** | measured, current source; **confirmed to +0.2 % by the real run** |
| RTX 4090 | ~204 GPU-h | ~815 GPU-h | derived from the paired ratio ×2.67 |
| RTX 5090 | ~174 GPU-h | ~695 GPU-h | derived from the paired ratio ×3.13 |

**About 2,000 GPU-hours on a 4070 Super; ~800 on a 4090.** For comparison, n=31
consumed 498 on the rented cards against 497 predicted before renting — it is
that 0.2 % agreement that lends confidence to the ×4 projection.

The first two rows are measured with the very binary that produced the result, on
the same fixed-seed sample of 96 `vhi`. The last two are not: their absolute
values predate v7, and only the card-to-card **ratio** is considered stable (§7.1
of [TRACE.md](TRACE.md)). They are therefore obtained by applying that ratio to
the current 4070 measurement, and should be re-measured before committing to a
rental.

Wall-clock time is just a division: the same fleet of 32 × 4070 Super would do it
in **62 h**, 128 cards in 16 h. The dollar cost is likewise fleet-independent —
**≈ $190** at $0.0945/GPU-h, the rate actually paid here. n=32 is therefore
**immediately within reach**.

### What it would take in code — three real obstacles

**1. The kernel is not instantiated for 32.** `pick()` and `dpick()` only cover
{9…24, 27, 28, 31}. `INST(32)` and `DINST(32)` must be added.

**2. A silent overflow in the row mask.** Rows are `uint32_t` and the code
computes:

```c
const uint32_t ALLB0 = (1u<<N)-1u;      // N=32  ->  1u<<32
```

Shifting a 32-bit type by 32 is **undefined behaviour** in C/C++. On current
hardware it yields 1 (shift modulo 32), so `ALLB0` would be 0 and row negation
would be silently wrong. This needs either a special case or a wider
representation. **n=32 is the last n that fits in a `uint32_t`**: 32 cells per
row, 31 free bits after pinning.

**3. The accumulator margin tightens.** The total is V·2^{2n}, so:

| n | log2(V) | + 2n | bits used / 160 | margin |
|---|---|---|---|---|
| 31 | 83.3 | 62 | 145.3 | 14.7 bits |
| 32 | 87.3 | 64 | **151.3** | **8.7 bits** |

It fits, but only ~8.7 bits remain. For n=35 (the next n ≡ 3 mod 4) a 160-bit
accumulator would be insufficient.

The real wall remains the 4ⁿ: each further n costs 4×, and n=35 would cost 256×
n=31 — roughly 127,000 GPU-h and ~$12,000. §5 of [TRACE.md](TRACE.md) argues that
breaking that exponent would require an unknown mechanism.

## Cost of the experiment

| item | amount |
|---|---|
| compute — 15.7 h × $3.044/h (4 machines × 8 RTX 4070 Super) | $47.74 |
| aborted rentals (a preempted 14×4090, a fraudulent 8×4090, tunnel retries) | ≈ $2 |
| local RTX 4070 | electricity |
| **total** | **≈ $50** |

The computation consumed **498 GPU-h** on the rented cards against the **497 h**
predicted by the benchmark before renting: a **+0.2 %** discrepancy. Load stayed
balanced across the four machines (2,027 / 1,998 / 1,970 / 1,957 tasks), and the
local 4070 absorbed 238.

One lesson was paid for in cash: the cheapest offer on the market, an "8× RTX
4090" at $1.72/h, **faked `nvidia-smi`** through an `/opt/fake/nvml_reader.py`
and ran no sm_89 kernel at all. The discount was the symptom. The orchestrator
now rejects such hosts automatically and accepts a machine only after making it
compute L(2,12) = 108144.

## AI-authored work

**This entire repository was produced by Claude Opus 5** (Anthropic), from the
first commit onward. All 40 commits carry `Co-Authored-By: Claude Opus 5`, across
three sessions between 4 and 6 September 2026.

That covers: deriving the parity decomposition and its proof, the CUDA kernel and
its PTX assembly, the fourteen research avenues explored and then closed
(including the two negative results above), the distributed orchestrator and its
preemption tolerance, the verification chain, renting and driving GPUs at an
external provider, and this page.

The human role was to set the objectives, provide access to the hardware and the
rental account, and arbitrate spending decisions.

What was **not** delegated to the AI: verification by third parties. That is
precisely why the [Verifying the result yourself](#verifying-the-result-yourself)
section exists and why every partial sum carries its provenance. A result
produced by an AI does not need independent recomputation any less than one
produced by a human — it needs it more.

## References

See the [French section](#références) above; the bibliography is identical.

## Files

| file | role |
|---|---|
| `langford6.cu` | the CUDA kernel, all optimizations |
| `slice_ref.c` | the reference implementation, a *different* algorithm |
| `orchestrator.py` | renting, lease-based distribution, preemption recovery |
| `finalize.py` | closing out: preserve, release machines, prove |
| `dashboard.py` | progress at `http://localhost:8800` |
| `audit.sh` | the audit dossier, redundant recomputation included |
| `check_refs.sh` | cross-check of the two algorithms |
| `verify_all.sh` | full chain on small n |
| `estimate.c` | Knuth's estimator, independent yardstick |
| **[TRACE.md](TRACE.md)** | **the complete lab notebook, 2,300 lines** |
