# Langford L(2,n) — comptage exact sur un seul GPU

Compter **toutes** les suites de Langford d'ordre *n* : arranger 1,1,2,2,…,n,n de
sorte que les deux copies de *k* soient séparées par exactement *k* autres
nombres. Exhiber une solution est trivial ; les compter toutes est un problème
ouvert au-delà de n=28.

Ce dépôt contient une implémentation CUDA de la méthode algébrique de Godfrey,
poussée jusqu'à **5,30× plus vite** que mon point de départ, avec au passage la
fermeture — par preuve ou par mesure — de trois pistes qui étaient jusque-là
seulement « non abouties ».

| | n=24 | n=27 | n=28 | **n=31** |
|---|---|---|---|---|
| point de départ (v3) | 15,9 min | 16,9 h | 2,82 j | 180,1 j |
| **v6 (ce dépôt)** | **3,0 min** | **3,18 h** | **12,7 h** | **≈ 34,0 j** |

**Ce qui est fait** : le record publié de 2015 — L(28), obtenu sur ~32 GPU en
9 jours — se refait ici en **12,7 heures sur une seule RTX 4070**.
**Ce qui ne l'est pas** : L(2,31) reste inconnu. Ce dépôt n'a pas battu le
record ; il en a divisé le coût par ~545.

---

## 1. Le problème, et ce que « battre le record » veut dire

Deux problèmes très différents portent le même nom.

| | difficulté |
|---|---|
| **Exhiber** une suite de Langford pour n=31 | trivial — `./find_langford 31` répond en < 1 ms |
| **Compter** toutes les suites L(2,31) | **c'est le record ouvert** |

Une suite existe ssi n ≡ 0 ou 3 (mod 4), donc après n=28 vient **n=31** : le plus
petit cas non résolu. Valeurs connues : n ≤ 24 (Krajecki *et al.*, 2004-05),
n = 27 et 28 (Assarpour, Bar-Noy & Liu, 2015, [arXiv:1507.00315]).

Estimations indépendantes pour L(2,31) : **5,74·10²⁴ ± 2 %** (`estimate.c`,
estimateur de Knuth non biaisé), 5,70·10²⁴ (Zan Pan, 2021), 5,3·10²⁴ (Assarpour
*et al.*, trempe parallèle). Les énumérer une par une est hors d'atteinte
(~2·10⁵ ans à 10¹² solutions/s) : seule une méthode algébrique est viable.

---

## 2. La méthode

### 2.1 Godfrey (2002)

Une suite de Langford est un couplage parfait des 2n positions dont le multi-
ensemble des écarts est exactement {2, 3, …, n+1}. Godfrey encode cela par

    F(n,X) = prod_{i=2}^{n+1} A_i(X),   A_i(X) = sum_{k=1}^{2n-i} x_k x_{k+i}

Le nombre de couplages est le coefficient de x₁x₂…x_{2n}. F est homogène de
degré 2n en 2n variables, donc tout monôme non multilinéaire possède une
variable d'exposant pair : la substitution ±1 le tue. D'où

    V(n) = 2·L(2,n) = 2^{-2n} · SUM_{X in {-1,1}^{2n}} (prod_k x_k) · F(n,X)

Coût **Θ(4ⁿ)**. C'est l'état de l'art publié, inchangé depuis 2002.

### 2.2 Décomposition de parité — la clé de la v6

Séparons les positions **impaires** (rangée `o`, n cases) des **paires**
(rangée `e`, n cases). Un calcul d'indices direct donne, pour tout n :

    ecart PAIR   i = 2m   :  A_i = P_m(o) + Q_m(e)          (rangees SEPAREES)
    ecart IMPAIR i = 2m+1 :  A_i = sum_j O_j E_{j+m} + sum_j E_j O_{j+m+1}   (bilineaire)

où P_m, Q_m sont les autocorrélations de rangée au lag m. Vérifié exactement sur
6,2 millions d'écarts jusqu'à n=31 (`oe_check`, 0 divergence).

Deuxième observation : A_i est une somme de (2n−i) termes ±1, donc
**A_i ≡ i (mod 2)** — seuls les écarts **pairs** peuvent s'annuler. Or c'est
l'annulation d'un facteur qui décide qu'un point ne contribue pas.

**Conséquence exploitée par la v6** : la survie d'un point ne dépend de `o` que
par le vecteur constant P(o). Pour un `e_hi` fixé, la carte `e_lo → Q(e)` ne
dépend pas du tout de `o` et se tabule une fois pour tout un bloc de threads.

### 2.3 Le groupe de symétries

En coordonnées de parité, le groupe de Klein {id, ν, ε, νε} devient la
**négation indépendante de chaque rangée** — il agit simplement transitivement
sur (o₁, e₁), donc on épingle o₁ = e₁ = 0 et on multiplie par 4. La réflexion
ρ : p ↦ 2n+1−p **échange les deux rangées** :

    sigma(o,e) = (f(e), f(o))     avec  f = canon o rev

(rev = renversement des n bits, canon = négation de la rangée si son bit 0 vaut
1). `f` est une involution, donc le représentant canonique est `e ≤ f(o)`, et en
reparamétrant le thread par F (avec o = f(F), licite puisque f est involutive)
le prédicat devient **v ≤ u** : les blocs sans travail ne sont jamais lancés.

---

## 3. Pourquoi le résultat est valide

### 3.1 Ce qui est démontré

1. **L'identité de Godfrey.** F homogène de degré 2n en 2n variables ⟹ la somme
   ±1 ne retient que les monômes à exposants tous impairs ; somme des exposants
   = 2n avec 2n exposants impairs ⟹ tous valent 1. Le coefficient multilinéaire
   compte les systèmes ordonnés de cordes couvrant chaque position une fois,
   soit 2·L(2,n).
2. **La décomposition de parité** (§2.2) : calcul d'indices, plus vérification
   numérique exhaustive par échantillonnage jusqu'à n=31.
3. **A_i ≡ i (mod 2)**, donc seuls les 16 écarts pairs peuvent s'annuler : c'est
   la correction du test de survie de la v4/v5/v6.
4. **Le groupe de symétries est d'ordre exactement 8.** Un motif de signes ε avec
   A_i(εx) = ±A_i(x) impose ε_k·ε_{k+i} constant en k pour tout i, donc
   e_{k+1} − e_k constant dans F₂ : identité, négation globale, alternance.
   Avec la réflexion (seule permutation des positions préservant toutes les
   distances), le groupe est {id, ν, ε, νε} × {id, ρ}, d'ordre 8. Confirmé
   expérimentalement : `fiber` mesure les fibres de b ↦ (A₂,…,A_{n+1}), dont le
   stabilisateur connu est d'ordre 4 — **98 % des fibres font exactement 4**
   (n=11 : 1 018 688 fibres de taille 4 contre 14 018 de taille 8), les rares
   plus grandes étant des coïncidences isolées sans structure de groupe.
5. **La troncature à 160 bits est exacte.** La somme vraie vaut 2^{2n}·V(n)
   ≈ 2^{145,3} < 2¹⁶⁰ : l'arithmétique modulo 2¹⁶⁰ *est* la réponse exacte, avec
   15 bits de marge.
6. **La réflexion en coordonnées de parité** : `oe_ref2.c` énumère la moitié
   canonique `e ≤ f(o)` (poids 2) plus les orbites fixes `e = f(o)` (poids 1) et
   reproduit exactement n = 7, 8, 11, 12.

### 3.2 Validation empirique

**Toutes les valeurs connues sont reproduites exactement par la v6** :

| n | L(2,n) | v6 |
|---|---|---|
| 11, 12 | 17 792 ; 108 144 | ✓ |
| 15, 16 | 39 809 640 ; 326 721 800 | ✓ |
| 19, 20 | 256 814 891 280 ; 2 636 337 861 200 | ✓ |
| 23 | 3 799 455 942 515 488 | ✓ |
| 24 | 46 845 158 056 515 936 | ✓ |

(n = 7, 8 sont hors du domaine de découpage de la v6 ; v3/v4/v5 les reproduisent.)

Quatre contrôles indépendants s'ajoutent :

* **Recoupement croisé entre versions.** v3, v4 et v5 produisent des sommes
  partielles **identiques au bit près** sur n=31, pour deux géométries de shard
  indépendantes (t=12 shards 0..59 ; t=10 shards 3000..3024).
* **Recoupement par changement complet d'énumération.** La v6 n'a en commun avec
  les précédentes ni les coordonnées, ni l'ordre de parcours, ni la mise en
  œuvre de la symétrie, ni la structure de boucle. Qu'elle retrouve exactement
  les mêmes valeurs jusqu'à n=24 est une vérification indépendante forte.
* **Auto-test intégré, gratuit.** Le total doit être divisible par 2^{2n}
  (= 2⁶² pour n=31) : les 2n bits de poids faible doivent sortir nuls. Une
  corruption quelconque échoue à ce test avec probabilité 1 − 2^{−2n}.
* **Référence CPU indépendante.** `oe_ref` recalcule V(n) dans les coordonnées
  de parité, sans GPU, et concorde.

### 3.3 Ce que la validation ne couvre pas

Elle ne prouve pas l'absence de bug qui ne se manifesterait qu'à n=31 : les
chemins de code dépendants de n (largeurs de masques, nombre d'écarts, tailles
de table) sont exercés à n=24 mais pas aux valeurs exactes de n=31. L'auto-test
de divisibilité reste la garantie de dernier recours sur un run réel.

---

## 4. Les optimisations, dans l'ordre, avec les gains mesurés

Tous les débits sont mesurés GPU au repos, moyennés sur des shards répartis
(voir §4.6 : mesurer au mauvais endroit m'a coûté plusieurs heures).

| version | Gsums/s | n=31 | gain |
|---|---|---|---|
| v3 — Godfrey + symétrie ×8 | 37,0 | 180,1 j | — |
| v4 — saut des termes nuls | 65,6 | 101,7 j | ×1,77 |
| v5 — demi-état + déroulage | 89,0 | 75,0 j | ×2,40 |
| **v6 — parité + bitmaps de survie** | **392,7\*** | **34,0 j** | **×5,30** |

\* la v6 énumère 2^{2n−2} points nominaux dont la moitié n'est jamais lancée ;
le débit est rapporté à ce total nominal.

### 4.1 v4 — ne pas calculer les produits nuls (×1,77)

Mesure décisive : en remplaçant l'arbre de produit par une addition triviale, le
noyau passe de 37 à 126,5 Gsums/s — **le produit 160 bits est 70,6 % du temps**.
Et 87,1 % de ces produits valent zéro (§2.2 : seuls les 16 écarts pairs peuvent
s'annuler ; `zfrac` mesure P(aucun nul) = 12,92 % à n=31).

Un test par élément ne sert à rien — P(les 32 voies d'un warp mortes) = 0,87³²
= 1,2 %. La solution est de **compacter les survivants** dans une file
circulaire par warp en mémoire partagée, vidée par paquets de 32 : chaque
produit tourne alors sur un warp plein. Deux détails rendent le test bon marché :
les 16 écarts pairs sont regroupés dans 4 mots SWAR (expanseur de pas 2
`((v>>s)&0x55)*0x104104 & 0x04040404`, même nombre d'instructions), et le signe
du terme voyage dans la voie 31 inutilisée sous forme ±1.

### 4.2 v5 — ne suivre que la moitié de l'état (×1,36 de plus)

Seuls les écarts pairs décident de la survie ; les 15 impairs ne servent qu'à la
valeur du produit, soit 12,9 % du temps. La boucle chaude ne maintient donc que
les 16 pairs (4 registres au lieu de 8) et reconstruit les impairs depuis `b` au
moment du drain. Plus un déroulage par 8 : j = ctz(t+1) vaut 0,1,0,2,0,1,0,dyn
sur un bloc de 8 pas, donc 7 pas sur 8 adressent la banque de constantes avec un
décalage littéral.

### 4.3 v6 — supprimer la boucle chaude (×2,21 de plus)

C'est le saut structurel. Grâce à la décomposition de parité (§2.2), on
précalcule pour chaque écart *m* et chaque valeur atteignable *v* le **bitmap
des e_lo tels que Q_m(e_lo) = v**. Un thread obtient alors la mortalité de 32
points d'un coup par un OU de 16 mots : le test de survie tombe de ~65
instructions par point à **moins d'une**. Il n'y a plus de code de Gray du tout.

La réflexion reparamétrée (§2.3) rend le prédicat canonique `v ≤ u`, ce qui
permet de **ne pas lancer** les blocs sans travail.

### 4.4 Creuser le drain

Une fois la boucle chaude supprimée, tout est dans le drain. Décomposition
mesurée : produit 64 %, écarts impairs 55 %, base 30 % (les parts se recouvrent,
le noyau masquant de la latence).

* Les 15 écarts impairs sortent des popcounts en **entiers**, et on les
  empaquetait en octets SWAR pour que l'arbre les ré-extraie aussitôt. On les
  **multiplie deux à deux à la volée** : 8 produits vivants au lieu de 15
  entiers. −65 instructions par survivant.
* Le terme diagonal (orbites fixes `e = f(o)`, une par thread sur tout le run)
  part dans son propre noyau ; les 16 offsets de tranche sont empaquetés deux
  par registre. 48 registres au lieu de 64, 0 spill.
* **Tout en complément à deux modulo 2¹⁶⁰** : l'accumulateur étant déjà
  modulaire et |somme| < 2¹⁵⁹, ni valeur absolue ni suivi de signe ne sont
  nécessaires, avec pour seule correction
  `a_s·b_s = a_u·b_u − 2^w(s_a·b_u + s_b·a_u)`. −24 instructions et une branche.

### 4.5 Les corrélations que `e_lo` ne touche pas

R(m) = Σⱼ Oⱼ E_{j+m} porte sur les indices E de m+1 à N, et `e_lo` n'occupe que
les cases 2..K+1. Donc **pour m ≥ K+1 cette corrélation ne dépend pas du tout de
`e_lo`** : elle est constante sur tout le bloc interne. Avec K=7 cela concerne
8 des 15 écarts, soit 8 des 30 popcounts du drain. On les calcule une fois par
(thread, e_hi) et on les range en int16 dans `sPw`. Gain mesuré : **+4,4 %**.

Balayage de K : plus K est petit, plus de corrélations deviennent constantes,
mais moins la construction des tables s'amortit. K=7 reste l'optimum.

Drain final : **287 instructions SASS par point survivant**.

### 4.6 Essayé, mesuré, sans gain

* **Trafic de file vectorisé 128 bits** (`STS.128`/`LDS.128`) : la file n'est pas
  le goulot.
* **Équilibrage ALU/FMA** en forçant les IADD3 sur le pipe FMA via `mad.lo.u32` :
  ptxas les reconvertit — il a déjà fait ce choix, le noyau est limité par le
  débit d'émission, pas par le pipe ALU.
* **Drapeaux ptxas** (`-allow-expensive-optimizations`, `--opt-level=3`,
  `--extra-device-vectorization`) : 0 % à l'arrondi près.
* **Taille de bloc** 128 et 512 threads : −10 % et −3 %. 256 est l'optimum.
* **Occupancy 4 → 5 blocs/SM** (48 registres, 0 spill) : identique au dixième de
  jour près. Ce n'était pas le facteur limitant.
* **Accumulateur 128 bits.** Chaque terme est divisible par 2¹⁶ (les 16 facteurs
  pairs sont pairs), donc la somme tiendrait sur 4 limbes ; mais diviser par 2 en
  SWAR coûte 8 à 12 instructions et l'étage haut n'en économise que ~7.
* **Corrélation par multiplication entière.** Les 30 popcounts sont une
  convolution, donc un produit de polynômes empaquetés : il faut 6 bits par
  coefficient et 61 coefficients, soit 6×6 = 36 multiplications larges contre
  30 popcounts. Perdant.
* **Fusion des deux popcounts par écart.** On n'a besoin que de leur somme
  (C_m = 2N−2m−1 − 2(R+R')), donc un `__popcll` sur un mot 64 bits assemblé
  suffirait — sauf que `__popcll` est déjà compilé en deux POPC et une addition.

### 4.7 Quatre pièges de mesure, et une leçon d'architecture

Ils m'ont coûté plusieurs heures et sont reproductibles :

1. **Le shard 0 est dégénéré.** Les premières mesures de la v6 donnaient
   39 Gsums/s, moitié moins que la v5. Cause : à `vhi = 0` la rangée `e` est
   quasi nulle, les autocorrélations sont extrêmes et **87,9 %** des points
   survivent au lieu de 12,9 % — le drain tournait sept fois trop souvent. Les
   mesures de la v5 sur les shards 0..59 étaient elles aussi biaisées vers le bas
   (75 contre 92 Gsums/s en moyenne réelle).
2. **La plage d'échantillonnage.** Après l'ajout de la réflexion je n'ai mesuré
   que le premier quart de la plage de `vhi`, là où presque aucun thread ne
   saute : gain apparent 1,22× au lieu du vrai 1,9×. Le débit va de 237 à
   3 806 Gsums/s selon `vhi` ; seule l'intégrale sur toute la plage a un sens.
3. **La contention.** Une validation en tâche de fond fausse toute mesure de
   débit. Vérifier `nvidia-smi` avant chaque campagne.
4. **Les `vhi` de faible poids de Hamming sont dégénérés.** `vhi = 2²²` donne un
   `e_hi` presque nul, donc les mêmes autocorrélations extrêmes qu'à `vhi = 0` :
   le même binaire mesure 141 Gsums/s en `vhi = 4 194 304` et 441 en
   `vhi = 4 200 000`. Échantillonner des points isolés est donc invalide — il
   faut moyenner sur des **fenêtres de `vhi` contigus**. Toutes les mesures de ce
   README sont refaites avec ce protocole (fenêtres de 25 à 32, six positions
   réparties) ; il donne des chiffres ~15 % moins flatteurs que les points
   isolés, et reproductibles à 0,3 % près sur la v3.

Corollaire : la v3 est facile à mesurer parce que son coût est uniforme (elle
calcule tous les produits) ; à partir de la v4 le coût dépend des données, donc
la mesure devient un problème en soi.

Et la leçon : **l'empaquetage SWAR n'est pas du gaspillage, c'est de la
compression de registres.** Passer les 15 écarts impairs à l'arbre sous forme
d'entiers économise ~65 instructions sur le papier ; mesuré, c'est **1,85× plus
lent** (376 → 203 Gsums/s), parce que garder 15 entiers vivants fait déborder le
fichier de registres. La version qui marche les multiplie deux à deux (§4.4).

### 4.8  Qu'est-ce qui limite vraiment le noyau ? (trois hypothèses, trois mesures)

À 34 jours, la v6 tourne à ~60 % du plafond d'émission de la carte. J'ai testé
les trois explications possibles ; deux sont fausses, et la troisième dit qu'il
n'y a plus qu'un levier.

**Hypothèse 1 — déséquilibre des pipes.** Sur Ada le pipe ALU a 64 voies par SM
et le pipe FMA 128. Le décompte SASS du drain donne **161 instructions ALU contre
95 FMA** : l'ALU coûte 80 cycles, le FMA 24 — le FMA dort à 30 %. Un décalage à
droite peut pourtant se faire sur le pipe FMA : `x >> m = __umulhi(x, 2^(32−m))`.
Appliqué aux 22 décalages des écarts impairs, le mix devient 139 ALU / 117 FMA
(vérifié : 22 `SHF.R.U32.HI` remplacés par 22 `IMAD.HI.U32`). **Mesure : 4 % plus
lent** (33,98 → 35,43 jours). Rejeté — et donc le noyau n'est pas au plafond ALU.

**Hypothèse 2 — latence des chaînes de dépendance.** Test direct : faire varier
le nombre de warps résidents.

| blocs/SM | warps/SM | Gsums/s |
|---|---|---|
| 2 | 16 | 288,2 |
| 3 | 24 | 287,8 |
| 5 | 40 | 287,4 |

**Identique à 0,3 % près.** Seize warps suffisent déjà à masquer toute la
latence : le noyau n'est pas latence-lié. (Corollaire pratique : l'occupancy est
un non-sujet ici, ce qui explique pourquoi aucun réglage de registres n'a jamais
rien donné.)

**Hypothèse 3 — mémoire partagée.** LSU à ~15 % d'utilisation. Rejeté.

**Conclusion.** Le noyau est à ~80 % du plafond d'émission que lui impose son
propre mélange d'instructions (une instruction ALU occupe son pipe deux cycles,
une FMA un seul). Il n'est donc ni à optimiser par l'occupancy, ni par les
pipes, ni par la mémoire : **seul le nombre d'instructions compte encore**. Et
il est à 15 % de son plancher, avec les 22 popcounts des écarts impairs
exactement au minimum et l'arbre de produit à ~1,4× du sien.

---

## 5. Pistes de recherche

### 5.1 Fermées par une preuve

**Kasteleyn / Pfaffien — définitivement mort.** En variables de Grassmann le
Pfaffien donnerait 2ⁿ·n³ au lieu de 4ⁿ : le record tomberait en minutes. Le
piège est le signe : un Pfaffien compte Σ_M (−1)^{cr(M)}. `parity` montre que la
parité des croisements n'est pas constante — mais **un Pfaffien n'a jamais eu
besoin de cela**. L'extraction du monôme de couleurs ne retient déjà que les
appariements arc-en-ciel ; il suffit d'une pondération d'arêtes w telle que
cr(M) + Σ_{e∈M} w_e soit constante *sur les appariements de Langford*. C'est un
système linéaire sur GF(2) à |E| inconnues, et il n'avait jamais été testé.
`pfaff_test` le résout exactement :

| n | arêtes | appariements | rang | lignes impossibles |
|---|---|---|---|---|
| 3, 4 | 9, 18 | 2, 2 | 2, 2 | **0** |
| **7** | 63 | 52 | 41 | **2** |
| 8 | 84 | 300 | 62 | 126 |
| 11 | 165 | 35 584 | 134 | 17 616 |
| 12 | 198 | 216 288 | 164 | 107 808 |

cr(M) n'est donc pas une fonction affine de l'ensemble d'arêtes : **aucune
orientation ne corrige le signe.** Et ±1 n'est pas la seule option — Kasteleyn se
généralise au cercle unité : avec w_e = ζ^{t_e}, ζ = exp(2iπ/2^k), la condition
devient un système sur **Z/2^k**, *strictement plus faible* (il ne peut échouer
que sur les relations **entières** entre vecteurs d'appariement, pas sur les
relations mod 2). `pfaff_zk` le résout par élimination 2-adique : **même échec,
même rang, même nombre de lignes impossibles pour k = 1, 2, 3, 8 et 32.** À
k = 32 c'est U(1) à toutes fins pratiques.

> Aucune pondération de Kasteleyn n'existe, ni réelle ni complexe. C'était le
> seul chemin connu sous le 4ⁿ pour ce problème.

**Le groupe de symétries est d'ordre exactement 8** (§3.1, preuve + mesure des
fibres) : il n'y a pas de facteur 16 caché.

**Deux points d'évaluation par variable est le minimum.** La substitution ±1 ne
marche que grâce à l'homogénéité de degré 2n = nombre de variables ; grouper des
variables perd de l'information, et une substitution de Kronecker fait exploser
le degré.

### 5.2 Fermées par une mesure

**Meet-in-the-middle / DP (fenêtre, couleurs).** L'estimation classique du
nombre d'états à la coupe médiane, Σ_b C(n,b)·C(n,(n+b)/2), ignore la contrainte
de Hall liant fenêtre et couleurs. `dpstates` compte les états *réellement
atteignables* :

| n | pic d'états | transitions totales | log₂(pic)/n |
|---|---|---|---|
| 8 | 1 238 | 8 327 | 1,284 |
| 11 | 64 737 | 6,3·10⁵ | 1,453 |
| 12 | 240 697 | 2,7·10⁶ | 1,490 |
| 15 | 1,24·10⁷ | 2,0·10⁸ | 1,571 |
| 16 | 4,68·10⁷ | 8,4·10⁸ | 1,592 |

log₂(pic) croît de **1,90 par unité de n** (régulier de n=8 à n=16), log₂(des
transitions) de **2,07**. Extrapolé à n=31 : pic ≈ 2⁵⁴ états (la contrainte de
Hall ne gagne rien) et 2⁶¹ transitions — un **exposant pire que le 2n de
Godfrey**. La MITM n'est pas seulement bloquée par la mémoire, elle est aussi
asymptotiquement perdante.

**Björklund fusionné avec l'anneau de couleurs.** *Counting perfect matchings as
fast as Ryser* (SODA 2012, [arXiv:1107.4466]) donne O\*(2^{V/2}) couplages
parfaits, soit 2ⁿ pour nos 2n = 62 positions — exactement la borne de largeur
arborescente. Croisé avec la contrainte « chaque écart une fois » (un second
anneau nilpotent de dimension 2ⁿ), les dimensions se multiplient : on retombe
sur 4ⁿ avec des facteurs polynomiaux en plus.

**Largeur arborescente du graphe des positions.** Restreint à une fenêtre W de
n+2 positions consécutives, le graphe (p~q ssi 2 ≤ |p−q| ≤ n+1) a pour
complémentaire un **chemin** : deux positions de W sont non adjacentes ssi elles
sont à distance 1. Donc α(G[W]) = 2, et si S sépare G[W] en A et B non vides,
tout a ∈ A est non adjacent à tout b ∈ B, donc B est inclus dans les ≤ 2 voisins
de a dans le chemin : |A| ≤ 2 et |B| ≤ 2, d'où |S| ≥ |W| − 4. **La largeur
arborescente est ≥ n − 3** : le côté « positions » d'une DP vaut
irréductiblement 2ⁿ, le côté « couleurs » vaut 2ⁿ, et le produit est 4ⁿ.

**MacWilliams / distribution des poids de coupe** ([arXiv:1208.2329]). La somme
de Godfrey ne dépend de X que par les tailles de coupe c_i(X) dans les 31
graphes d'écart, donc par l'énumérateur de poids multiple d'un code linéaire de
longueur ~1000 et de **dimension 61**. MacWilliams renverrait au dual, de
dimension 939 : la transformation va dans le mauvais sens ici.

**Autres records de la famille.** L(3,n) est connu jusqu'à n=19, le cas ouvert
suivant est n=26 → 3n = 78 positions → 2⁷⁵ points, soit ~16 000 ans.
L(4,n) : seul n=24 est connu, l'ouvert suivant est n=31 → 4n = 124 positions.
Aucun record plus accessible dans la famille.

### 5.3 Ouvertes, non explorées à fond

* **Tensor cores.** Les 15 écarts impairs sont *bilinéaires* en (o,e), donc
  formellement un GEMM (~230 MAC par couple, contre ~1,16·10¹⁴ MAC/s en INT8).
  Cela retirerait ~40 % du drain du chemin ALU. Estimation non prototypée :
  plafond ALU restant ≈ 200 Gsums/s canoniques → ~20 jours. Le coût est une
  réécriture complète du drain avec sortie streamée en mémoire partagée.
* **Battre le 4ⁿ.** Reste ouvert. Les quatre voies connues sont fermées
  ci-dessus ; ce qu'il faudrait, c'est un mécanisme qui traite la contrainte de
  couverture (2^{2n} en inclusion-exclusion, 2ⁿ en largeur arborescente) et la
  contrainte de distinction des écarts (2ⁿ) **sans que les deux se multiplient**.
  La structure particulière du problème — la couleur d'une arête est déterminée
  par ses extrémités — n'a pas encore été exploitée pour cela.
* **Arithmétique du produit.** L'arbre 160 bits est à ~161 instructions contre un
  minimum théorique de ~116. Karatsuba au dernier étage, ou un ordonnancement
  qui réduit les 21 instructions de propagation de retenue, laisseraient peut-être
  10 à 15 %.

### 5.4  Idées propres, dérivées ici — et pourquoi elles ne cassent pas le 4ⁿ

Ces pistes ne viennent d'aucun article ; je les ai construites à partir de la
structure du problème. Aucune n'aboutit, mais chacune ferme quelque chose ou
éclaire pourquoi la barrière tient.

**(a) Une identité locale sur les croisements.** Pour une corde *c* de longueur
*d* dans un diagramme de cordes quelconque, il y a exactement *d−1* positions
strictement à l'intérieur. Chaque corde extérieure qui la croise y pose un pied,
chaque corde entièrement intérieure y en pose deux. D'où, exactement :

        #croisements(c) = d(c) − 1 − 2·#{cordes imbriquées dans c}

donc **#croisements(c) ≡ d(c) − 1 (mod 2)**. Vérifié sans exception sur tous les
appariements de Langford jusqu'à n=12 (`struct.c`). Conséquences :

* en sommant, **cr(M) ≡ nest(M) + K (mod 2)** avec K = Σ(d−1)/2 constant (K est
  pair pour n=31, donc cr ≡ nest) ;
* le graphe des croisements a une **suite de parités de degrés fixe** : ce sont
  exactement les 16 cordes de longueur paire qui croisent un nombre impair de
  cordes ;
* et, en comptant les cordes ouvertes au moment de chaque ouverture,
  **nest(M) + cr(M) = Σ_c o(a_c)** — quantité, elle, parfaitement suivie par une
  DP à fenêtre.

Cette dernière identité est frustrante de près : on sait suivre `nest + cr` en
2ⁿ, on connaît `cr − nest` **modulo 2**, mais pas sa valeur — donc on ne sépare
pas cr. Le signe du Pfaffien reste hors de portée.

**(b) Kasteleyn généralisé — et son certificat de décès.** Le §5.1 montre qu'il
n'existe pas de pondération d'arêtes rendant cr affine. Mais la vraie condition
pour qu'un Pfaffien fonctionne est plus faible : il suffirait que **cr se
factorise par *d* formes linéaires** de l'ensemble d'arêtes. On pourrait alors
faire le Pfaffien sur l'algèbre de groupe (Z/2)^d — de dimension 2^d — et lire
le signe dans chaque classe, pour un coût **2^{n+d}·poly**. Si *d* était petit,
le 4ⁿ tomberait.

Test : *d* doit au moins valoir log₂ du nombre de « différences mauvaises »
(paires d'appariements de parités de croisement différentes), sinon un
sous-espace de codimension *d* ne peut les éviter toutes.

| n | appariements | paires à cr différent | d ≳ | 2^{n+d} | 4ⁿ |
|---|---|---|---|---|---|
| 7 | 52 | 672 | 9,4 | 2^{16,4} | 2^{14} |
| 8 | 300 | 21 344 | 14,4 | 2^{22,4} | 2^{16} |
| 11 | 35 584 | 3,16·10⁸ | 28,2 | 2^{39,2} | 2^{22} |
| 12 | 216 288 | 1,17·10¹⁰ | 33,4 | 2^{45,4} | 2^{24} |

*d* croît comme **~2,8n**, donc 2^{n+d} ≈ 2^{3,8n} — bien pire que 4ⁿ = 2^{2n}.
La généralisation est morte, et pour une raison quantitative : le nombre
d'appariements de Langford croît en 2^{3,7n}, donc les parités de croisement
sont bien trop « désordonnées » pour tenir dans peu de bits.

**(c) Le vrai nom de la barrière.** Le compte de Langford est exactement le
coefficient multilinéaire d'un produit de *n* formes quadratiques,

        [x_1…x_{2n}]  prod_{i=2}^{n+1} (x^T M_i x)

c'est-à-dire un **hafnien mixte**. Son analogue *signé* — le **discriminant
mixte** — se calcule, lui, en 2ⁿ·poly par inclusion-exclusion sur les *n*
matrices : D(A_1..A_n) = (1/n!)·Σ_{S⊆[n]} (−1)^{n−|S|} det(Σ_{i∈S} A_i). La
barrière 4ⁿ ici **est** l'écart permanent/déterminant, ni plus ni moins, et
Kasteleyn en est le seul pont connu — pont dont j'ai montré (§5.1, et (b)
ci-dessus) qu'il n'existe pas pour Langford.

**(d) Une seule équation au lieu de 2n.** Si v est le vecteur de couverture,
Σ_k v_k = 2n est automatique, et **v = 𝟙 ⟺ Σ_k v_k² = 2n**. Tout le problème
tient donc en une équation quadratique. Mais Σv² = 2n + 2·(#collisions), donc
compter à Σv² fixé revient à compter par nombre de conflits — une fonction de
partition à interactions par paires sur un graphe complet de n couleurs. Aucun
gain : c'est l'inclusion-exclusion sur les positions qui en est déjà la forme
efficace.

**(e) Réécriture polynomiale univariée.** En notant s_i la position d'ouverture
de la couleur i, le problème est exactement

        sum_{i=2}^{n+1}  x^{s_i} (1 + x^i)  =  x + x² + … + x^{2n}

une identité de polynômes. Évaluée modulo x^{2n+1}−1 elle devient une identité
dans l'algèbre du groupe cyclique, donc une condition sur des racines de
l'unité. Séduisant, mais compter les solutions d'une identité polynomiale
ramène à extraire des coefficients, c'est-à-dire à Godfrey.

**(f) Repliement multi-modules.** Modulo x^d−1, la condition ne porte que sur
les (s_i mod d) : une inclusion-exclusion sur d classes au lieu de 2n, soit
2^d. Nécessaire mais pas suffisant. Et combiner plusieurs modules ne suffit pas
non plus : connaître les sommes par classe modulo d₁ et modulo d₂ ne donne que
deux projections marginales d'un tableau bidimensionnel — c'est de la
tomographie discrète, structurellement sous-déterminée.

**(g) Version cyclique et spectre de puissance.** Pour la variante *cyclique*
du problème, les autocorrélations deviennent circulaires, donc (Wiener-Khinchin)
entièrement déterminées par le spectre de puissance |X̂(ω)|² — et le poids
(∏x_k) l'est aussi, car |Σx| est lu dans le spectre et sa parité est forcée. La
sommande ne dépend alors que du spectre. Malheureusement le nombre de spectres
distincts d'une suite ±1 vaut ~2^{2n}/(4n) : aucun gain exponentiel. Et de toute
façon Langford est aperiodique.

**(h) Hafniens de Toeplitz.** L'inclusion-exclusion sur les couleurs donne
exactement

        compte arc-en-ciel  =  sum_{S} (-1)^{n-|S|} · haf(A_S)

où A_S est la matrice d'adjacence du graphe des positions dont les écarts sont
dans S. Or **ces matrices sont de Toeplitz** : A_S[p][q] ne dépend que de p−q.
Si le hafnien d'une Toeplitz 0/1 était calculable en temps polynomial, le tout
tomberait à **2ⁿ·poly** — et le 4ⁿ avec.

Il y a une raison d'espérer : sur un chemin, la matrice de transfert de la DP à
fenêtre est **la même à chaque position**. Donc la suite N_S(k) = nombre de
couplages parfaits sur [1..k] satisfait une récurrence linéaire. Si son ordre
était petit, on calculerait quelques termes pour de petits k — où la fenêtre est
étroite et la DP bon marché — puis on extrapolerait jusqu'à k = 2n.

Mesure de l'ordre minimal par Berlekamp-Massey (`toeplitz.c`) :

| S | fenêtre | états | ordre de la récurrence |
|---|---|---|---|
| {2,3} | 3 | 8 | **8** |
| {2..4} | 4 | 16 | **16** |
| {2..5} | 5 | 32 | **32** |
| {2..6} | 6 | 64 | **64** |
| {2..7} | 7 | 128 | **128** |
| {2..8} | 8 | 256 | **256** |
| {2..9} | 9 | 512 | **512** |

**L'ordre vaut exactement le nombre d'états** : le polynôme minimal de la matrice
de transfert est de degré plein, il n'y a aucune compression à en tirer.
Extrapoler coûterait autant que dérouler la DP. Piste morte, et pour une raison
structurelle nette.

**(i) Grassmannisation partielle.** C'est l'idée la plus prometteuse que j'aie
eue, parce qu'elle attaque l'écart permanent/déterminant par le milieu au lieu
de chercher à le franchir.

Le gouffre 4ⁿ contre 2ⁿ·poly est exactement celui entre **variables nilpotentes
commutantes** (extraction en 2^{2n} évaluations ±1, couleurs gratuites — c'est
Godfrey) et **variables de Grassmann anticommutantes** (2ⁿ Pfaffiens, mais le
résultat est signé et le signe n'est pas rattrapable). Rien n'oblige à choisir :
on peut rendre **un sous-ensemble A de positions** grassmannien et laisser le
reste commutant. L'homogénéité tient encore (le degré en y vaut |Aᶜ| = le nombre
de variables y), donc l'extraction coûte 2^{|Aᶜ|} évaluations ±1, et pour chacune
l'intégrale de Berezin sur A demande l'inclusion-exclusion sur les n couleurs :

        coût = 2^{n + |Aᶜ|}

* |A| = 0 : 2^{2n} — Godfrey (avec évaluation directe, les couleurs redeviennent
  gratuites) ;
* |A| = 2n : 2ⁿ — le Pfaffien pur, mais signé, donc mort ;
* **|A| > n : sous le 4ⁿ.**

Il faut pour cela que `sign_A(M)·∏_{e∈M} w_e` soit constant sur les appariements
de Langford, où sign_A(M) est la parité du nombre d'inversions de la suite des
positions de A lues dans l'ordre des couleurs. C'est le même test GF(2) que le
§5.1, mais **restreint à A** — donc strictement plus faible, et il pourrait
passer là où Kasteleyn échoue.

Mesure (`partial.c`, glouton avec 40 redémarrages aléatoires) :

| n | 2n | seuil de gain | **|A| maximal** | coût | 4ⁿ |
|---|---|---|---|---|---|
| 7 | 14 | > 7 | **6** | 2^15 | 2^14 |
| 8 | 16 | > 8 | **5** | 2^19 | 2^16 |

Le maximum atteignable **décroît** (6 puis 5) alors que le seuil **croît** (8
puis 9) : l'écart se creuse. Les positions retenues sont d'ailleurs un préfixe
— {1,2,3,4,5} — c'est-à-dire exactement là où il y a peu de croisements. Piste
fermée, et sa forme dit pourquoi : le signe grassmannien est irréductiblement
global.

**(j) La contrainte de couleurs tient en UNE équation entière.** C'est le fait le
plus surprenant que j'aie trouvé. Soit mᵢ la multiplicité de la couleur i ; on a
toujours Σmᵢ = n. Alors

        sum_j 2^{d_j} = 2^{n+2} − 4     <=>     tous les m_i valent 1

*Preuve.* Parmi toutes les écritures d'un entier N comme Σmᵢ2ⁱ avec mᵢ ≥ 0, la
représentation **binaire minimise la somme des chiffres** (chaque report
2^{i+1} → 2·2ⁱ l'augmente de 1). Or 2^{n+2}−4 = 2²+2³+…+2^{n+1} a exactement n
chiffres à 1, donc l'écriture de somme de chiffres n est unique. ∎
Vérifié exhaustivement jusqu'à n=12 (`single.c`, 1 352 078 multi-ensembles à
n=12, zéro contre-exemple).

**Les n bits « quelles couleurs sont utilisées » se réduisent donc à un seul
scalaire** — qu'on impose par transformée de Fourier, pendant que le comptage
pondéré des couplages se fait en 2ⁿ (Björklund). Coût = *(portée de la
statistique)* × 2ⁿ.

Mieux : **quatre sommes de puissances suffisent aussi.** Nombre de
multi-ensembles partageant p₁…p_k avec l'ensemble plein (`powersums.c`) :

| n | k=1 | k=2 | k=3 | **k=4** |
|---|---|---|---|---|
| 9 | 910 | 25 | 3 | **1** |
| 11 | 9 686 | 178 | 10 | **1** |
| 13 | 110 780 | 1 403 | 33 | **1** |
| 15 | 1 328 980 | 11 307 | 129 | **1** |

**Pourquoi ça ne suffit quand même pas.** Toute statistique additive séparante g
doit distinguer les sommes de tous les sous-ensembles de **même taille** (les
δ = 1_A − 1_B avec |A| = |B| sont des directions interdites). Cela fait C(n,n/2)
sommes distinctes dans [0, n·N], où N est la portée, d'où

        N ≥ C(n, n/2) / n ≈ 2ⁿ / n^{1,5}

et un coût total **≥ 4ⁿ/poly** : la route ne peut pas descendre sous le 4ⁿ
autrement que d'un facteur polynomial. En pratique elle fait pire : Σ2^d a une
portée n·2^{n+1}, soit un coût 2^{68} pour n=31 contre 2^{62} pour Godfrey ; les
meilleures constructions connues d'ensembles à sommes de sous-ensembles
distinctes (Conway–Guy, Erdős) restent en Θ(2ⁿ/√n) ; et les quatre sommes de
puissances ont une portée n^{14}, bien pire encore.

C'est la formulation la plus nette de la barrière que j'aie trouvée : **la
contrainte de couleurs est un seul nombre, mais ce nombre a irréductiblement
n bits.**

**(k) Le mécanisme de Björklund, enfin lu.** Son 2^{V/2} vient d'une opération
« retirer-remplacer » : on retire les sommets **par paires** et on les remplace
par une **étiquette** unique, parce qu'un couplage partiel couvre toujours les
deux sommets retirés ensemble. 2n sommets donnent donc n étiquettes, les
couplages parfaits deviennent les couvertures exactes de ces n étiquettes, d'où
un anneau de dimension 2ⁿ et O\*(2ⁿ) **opérations d'anneau**. Sur notre anneau de
couleurs (dimension 2ⁿ), chaque opération coûte 2ⁿ : on retombe sur 4ⁿ. Les deux
univers — n étiquettes de positions, n couleurs — sont disjoints, et rien ne les
identifie. Björklund note lui-même que sa technique « fait un usage profond du
fait qu'elle compte des couvertures de 2-ensembles » et donne une borne
conditionnelle expliquant pourquoi elle ne s'étend pas aux 3-ensembles — or notre
problème est exactement une couverture exacte par 3-ensembles {p, p+i, couleur i}.

**(l) Godfrey est optimal dans son propre cadre.** Toute méthode « poids local »
calcule Σ_s ∏_p g(v_p), où v_p est le nombre de cordes couvrant la position p et
g encode la contrainte de couverture. Si g s'écrit comme somme de K
exponentielles, g(v) = Σ_{k=1..K} c_k λ_k^v, alors

    Σ_s ∏_p g(v_p) = Σ_{k : [2n]→[K]} (∏_p c_{k_p}) · ∏_i ( Σ_s λ_{k_s} λ_{k_{s+i}} )

— la somme sur les placements **factorise par couleur**, et c'est exactement ce
qui rend les couleurs gratuites chez Godfrey. Le prix est une somme sur K^{2n}
étiquetages, donc **coût = K^{2n}**.

Quel est le K minimal ? g doit valoir 1 en v=1 et annuler tout le reste, ce qui
sur {0,…,n} demanderait a priori n+1 exponentielles. Mais la contrainte
**globale** Σ_p v_p = 2n (n cordes, deux positions chacune) permet de se
contenter de g(v) = [v impair] : 2n exposants impairs de somme 2n valent tous 1.
Et [v impair] = (1 − (−1)^v)/2, soit **K = 2**, λ ∈ {+1, −1}. Une g avec K=1 est
de la forme c·λ^v, jamais nulle en v=0 sans l'être partout : K=1 est impossible.

**Donc K = 2 est le minimum, Godfrey l'atteint, et 4ⁿ est optimal dans ce cadre.**
En sortir demande de corréler les étiquettes entre positions — c'est la DP à
fenêtre, de largeur arborescente ≥ n−3, donc 2ⁿ — mais les couleurs cessent
alors d'être gratuites et coûtent 2ⁿ à leur tour. 4ⁿ des deux côtés.

---

### 5.5  Récapitulatif : quatre formulations de la barrière

Après avoir fermé quatorze voies, voici ce que je crois comprendre du 4ⁿ. Ce ne
sont pas quatre obstacles indépendants mais quatre visages du même.

1. **Permanent contre déterminant.** Le compte est un hafnien mixte ; son
   analogue signé, le discriminant mixte, se calcule en 2ⁿ·poly. Kasteleyn est
   le seul pont connu, et il est absent ici — démontré en ±1, en U(1) jusqu'à
   Z/2³², dans sa généralisation à d formes linéaires (d ≈ 2,8n), et en
   grassmannisation partielle (|A| ≤ 6 quand il en faudrait > n).
2. **Deux univers de n bits, disjoints.** Björklund réduit les 2n positions à n
   étiquettes ; les couleurs en font n de plus ; rien ne les identifie. La DP
   jointe a un état mesuré à 2^{1,90n} et des transitions à 2^{2,07n}.
3. **La contrainte de couleurs est un seul nombre à n bits.** Elle se réduit à
   Σ2^{d_j} = 2^{n+2}−4, mais toute statistique additive séparante a une portée
   ≥ 2ⁿ/n^{1,5}, donc l'imposer par Fourier coûte 2ⁿ — et 2ⁿ·2ⁿ = 4ⁿ.
4. **Godfrey est optimal dans le cadre des poids locaux**, avec K = 2 minimal.

La conclusion que j'en tire : casser le 4ⁿ demanderait un mécanisme qui traite la
couverture et la distinction des écarts **sans que les deux exponentielles se
multiplient**. La structure particulière du problème — la couleur d'une arête est
déterminée par ses extrémités — est la seule prise qui reste, et je n'ai pas su
en faire quelque chose.

---

## 6. Est-ce un vrai progrès sur l'état de l'art ?

Il faut séparer trois questions.

**Asymptotiquement : non.** L'algorithme reste Θ(4ⁿ). L'exposant n'a pas bougé
depuis Godfrey (2002), et le §5 explique pourquoi les voies connues pour le
casser sont fermées. Ce dépôt n'apporte aucune amélioration de complexité.

**Sur la même instance, en coût machine : oui, largement.** Sur L(28), le
dernier point du record publié :

| | matériel | temps | GPU-jours |
|---|---|---|---|
| Assarpour, Bar-Noy & Liu (2015) | ~32 GPU Kepler | ~9 jours | ~288 |
| **ce dépôt (v6)** | 1 × RTX 4070 | **12,7 h** | **0,53** |

soit **~545× moins de GPU-jours**. Une part revient au matériel : un GPU Kepler
de 2013 délivre ~1,3·10¹² opérations entières/s contre 7,3·10¹² pour une 4070
(≈ 1,5·10¹³ en comptant le pipe FMA), soit un facteur **6 à 11**. Le reste —
**environ 50 à 90×** — vient de l'algorithme et de l'implémentation : symétrie
d'ordre 8 complète, saut des 87 % de produits nuls, décomposition de parité,
bitmaps de survie, et une arithmétique 160 bits tronquée plutôt que du CRT
modulaire.

**Sur le record lui-même : non, pas encore.** L(2,31) reste inconnu. Ce qui a
changé, c'est son prix :

| | avant | maintenant |
|---|---|---|
| sur une 4070 | 180 jours | **34,0 jours** |
| sur une RTX 4090 | ~67 jours | **~12,6 jours** |
| en location grand public (~0,35 $/h) | ~560 $ | **~106 $** |
| en parallèle | ~26 GPU pendant une semaine | **~5 GPU pendant une semaine** |

Le record passe d'une allocation de calcul intensif à un budget individuel.
C'est un changement de **classe d'accessibilité**, pas de classe de complexité.

**Apport négatif à la littérature.** Trois barrières étaient jusqu'ici
« essayées sans succès » ; elles sont maintenant fermées : Kasteleyn par un
argument de rang sur GF(2) puis sur Z/2^k (§5.1), la meet-in-the-middle par une
mesure d'exposant (§5.2), le groupe de symétries par preuve et par mesure des
fibres. Cela ne fait pas avancer le calcul, mais cela évite à quelqu'un d'autre
de refaire ces chemins.

### Distance au plancher

| | n=31 |
|---|---|
| **v6, mesurée** | **34,0 j** |
| même code à 100 % d'émission (inatteignable) | 19,2 j |
| compte d'instructions à son minimum (~250/survivant), à 100 % | 17,1 j |

42 instructions par point canonique (0,129 × 287 de drain, plus ~5 de base),
196,4 Gsums/s canoniques → 8,25·10¹² instructions/s contre un plafond d'émission
de 1,46·10¹³ : **57 % du plafond de la carte**, et **15 % au-dessus du plancher
d'instructions**. Les 22 popcounts restants des écarts impairs sont exactement
au minimum (2 SHF, 2 LOP3, 2 POPC, 2 arithmétiques par écart non précalculé) ;
l'arbre de produit est à ~1,4× du sien.

Ce qui reste n'est ni l'occupancy, ni les pipes, ni la mémoire — les trois ont
été testés et éliminés (§4.8). Le noyau est à ~80 % du plafond que lui impose son
propre mélange ALU/FMA ; seul le nombre d'instructions compte encore.

---

## 6.bis  Provenance des méthodes

Pour être clair sur ce qui est emprunté et ce qui ne l'est pas.

### Repris de la littérature

| méthode | source | usage ici |
|---|---|---|
| **Méthode algébrique** F(n,X) = ∏ᵢ Aᵢ(X), extraction du coefficient multilinéaire par substitution ±1 | **M. Godfrey (2002)**. Pas de publication formelle ; décrite dans Assarpour–Bar-Noy–Liu §3 et dans Knuth, *TAOCP* vol. 4, pré-fascicule 5B (« Langford pairs ») | **c'est le cœur du calcul** — tout ce dépôt en est une implémentation |
| Mise en œuvre CUDA de Godfrey, symétries X → −X, X → reverse(X), alternance ; épinglage de deux coordonnées (facteur 4) | **Assarpour, Bar-Noy & Liu**, *Counting Skolem Sequences*, arXiv:1507.00315 (2015-17), §4 | point de départ des versions v1/v2 ; leur arithmétique modulaire + CRT a en revanche été **remplacée** par un bignum tronqué (mesuré ~4× moins cher) |
| Orientation pfaffienne : condition pour que sign(M)·∏ w_e soit constante sur les couplages | **P. W. Kasteleyn** (1961, 1963) ; formulation moderne dans Lovász–Plummer, *Matching Theory*, ch. 8 | testée **et réfutée** ici sur les appariements de Langford (§5.1), d'abord sur GF(2) puis sur Z/2^k |
| Intégrale de Berezin / variables de Grassmann : ∫∏Qᵢ = [t₂…t_{n+1}] Pf(Σ tᵢAᵢ) | standard en mécanique statistique (Zinn-Justin, ch. 1) | sert à formuler la piste pfaffienne du §5.1 |
| Couplages parfaits en O\*(2^{V/2}) par anneau nilpotent | **A. Björklund**, *Counting Perfect Matchings as Fast as Ryser*, SODA 2012, arXiv:1107.4466 | évalué (§5.2) : fusionné avec l'anneau de couleurs, les dimensions se multiplient et on retombe sur 4ⁿ |
| Hafnien en 2^{n−Ω(√n)} | Björklund, Kaski, Williams et suites, arXiv:2309.15422 (2023) | évalué : asymptotique pur, sans contrainte de couleurs, inutilisable à n=31 |
| Distribution des poids de coupe et identité de MacWilliams pour compter des couplages | *A New Direction for Counting Perfect Matchings*, arXiv:1208.2329 | évalué (§5.2) : ici le code est de dimension 61 et son dual de dimension 939, la dualité va dans le mauvais sens |
| Barrière 2^{\|U\|} pour le comptage de couvertures exactes | Set Cover Conjecture, **Cygan, Dell, Lokshtanov, Marx, Nederlof, Okamoto, Paturi, Saurabh, Wahlström** (2016) | encadre pourquoi 4ⁿ = 2^{2n} résiste (§5.2) |
| Estimateur Monte-Carlo non biaisé du coût d'un arbre de backtracking | **D. E. Knuth**, *Estimating the efficiency of backtrack programs*, Math. Comp. 29 (1975) | `estimate.c` — donne L(2,31) ≈ 5,74·10²⁴ ± 2 %, et sert à confirmer la marge de l'accumulateur 160 bits |
| Condition d'existence n ≡ 0, 3 (mod 4) et construction | **R. O. Davies** (1959) ; problème posé par **C. D. Langford** (1958) | `find_langford.c` |
| Records antérieurs n ≤ 24 | **Krajecki, Jaillet, Bui** *et al.* (2004-05), CONFIIT | valeurs de référence pour la validation |

### Techniques standard (folklore, non attribuables à un article)

* **Code de Gray binaire réfléchi** (Gray, 1953) : un seul bit bascule par pas,
  donc chaque autocorrélation ne bouge que de −4, 0 ou +4. Base des v1 à v5.
* **SWAR** (*SIMD Within A Register*, Fisher & Dietz 1998) : 31 autocorrélations
  en 8 registres de 4 octets, additions 32 bits sans propagation entre octets,
  garantie par une analyse d'intervalle.
* **Détection d'octet nul** `(v − 0x01010101) & ~v & 0x80808080` — *Bit Twiddling
  Hacks* (S. E. Anderson). Utilisée pour le test de survie ; j'ai vérifié que les
  faux positifs dus à l'emprunt n'apparaissent qu'en aval d'un octet réellement
  nul, donc que le prédicat reste exact.
* **Étalement de bits par multiplication** `(v & 0xF)·0x00810204 & 0x04040404`
  (et sa variante de pas 2 `(v & 0x55)·0x104104`) — même source ; j'ai vérifié
  l'absence de collision et de retenue sur les positions cibles.
* **Compaction de flux par warp** (`__ballot_sync` + rang, file en mémoire
  partagée) — technique NVIDIA classique des *warp-aggregated atomics*.
* **Multiplication multi-précision tronquée en complément à deux**, avec la
  correction a_s·b_s = a_u·b_u − 2^w(s_a·b_u + s_b·a_u) — Knuth, *TAOCP* vol. 2,
  §4.3.1.
* **Transformée de Walsh–Hadamard** : la somme de Godfrey *est* une convolution
  XOR de n fonctions indicatrices, ce qui rend explicite pourquoi elle coûte
  2^{2n} et pourquoi les fenêtres de temps n'y changent rien.

### Ce qui, à ma connaissance, est nouveau ici

Aucune de ces contributions n'est une avancée de complexité ; ce sont des
constantes et des résultats négatifs.

1. **Décomposition de parité exploitée comme séparation de variables** (§2.2).
   La décomposition « deux rangées » est connue (elle donne l'argument classique
   n ≡ 0, 3 mod 4) mais était considérée sans gain algorithmique. Le point neuf
   est que pour les écarts **pairs** — les seuls qui peuvent annuler le produit —
   les deux rangées sont *additivement séparées*, donc la survie d'un point ne
   dépend d'une rangée que par un vecteur constant. D'où les **bitmaps de
   survie**, qui suppriment complètement la mise à jour incrémentale d'état.
2. **Réflexion reparamétrée** (§2.3) : en coordonnées de parité la réflexion
   échange les rangées, σ(o,e) = (f(e), f(o)) avec f involutive ; en indexant le
   thread par F au lieu de o, le prédicat canonique devient v ≤ u, donc les blocs
   sans travail ne sont jamais lancés.
3. **Réfutation de Kasteleyn pour Langford** (§5.1), sous la forme faible et
   correcte du problème (pondération d'arêtes, pas parité constante), sur GF(2)
   **et** sur Z/2^k jusqu'à k=32.
4. **Mesure de l'exposant de la meet-in-the-middle** (§5.2) : 2,07n, contre une
   estimation antérieure qui la croyait seulement bloquée par la mémoire.
5. **Saut des produits nuls par compaction warp** (§4.1) : l'observation que
   87 % des produits sont nuls est ancienne, mais elle était rejetée comme
   inexploitable en SIMT ; la compaction la rend exploitable.

---

## 7. Budget pour n=31, et sur quelle carte

### 7.1  Ce qui est mesuré, et comment

Le coût d'une valeur de `vhi` varie d'un facteur **409** entre la plus rapide et
la plus lente (144 tirages) : l'élagage par les bitmaps de survie dépend du motif
de bits, pas de la magnitude. Trois points choisis à la main donnent donc
n'importe quoi — c'est exactement le piège de mesure n° 4 du §4.7. Le seul
estimateur honnête est un **tirage uniforme** de `vhi`, ce que fait `--bench` :

```
$ ./langford6 -n 31 --bench 64
GPU        : NVIDIA GeForce RTX 4070  (46 SM, sm_89, 2505 MHz)
echantillon: 64 vhi (graine fixe)  moyenne 0.3338 s/vhi  err-std 8.0%
PROJECTION n=31 : 777.7 h GPU = 32.40 jours   (IC95% 653.6 .. 901.8 h)
```

Contrôle : un échantillon hors-ligne de 144 tirages donne 772 h (IC95 %
696–850). Les deux méthodes concordent à 0,8 %.

> **RTX 4070 : n=31 en 32,2 jours (772 h GPU), IC95 % [696, 850].**
> Le chiffre de 34,0 j annoncé plus haut venait du débit nominal en Gsums/s ;
> la mesure directe est un peu meilleure. C'est celle qui fait foi.

La graine du tirage est **fixe**. Deux GPU différents mesurent donc le même
échantillon de `vhi` : la comparaison est appariée, et le rapport entre deux
cartes est bien plus précis que chacune des deux estimations absolues.

| n | v3 | v4 | v5 | **v6** |
|---|---|---|---|---|
| 24 | 15,9 min | 9,0 min | 6,6 min | **3,0 min** |
| 27 | 16,9 h | 9,5 h | 7,0 h | **3,18 h** |
| 28 | 2,82 j | 1,59 j | 1,17 j | **12,7 h** |
| **31** | 180,1 j | 101,7 j | 75,0 j | **32,2 j** (mesuré) |
| 32 | 1,97 an | 1,11 an | 0,82 an | **≈ 129 j** |

Tout l'état de l'art publié (L(27) + L(28)) se refait en **16 heures** sur cette
carte.

### 7.2  Quelle architecture, et pourquoi

Le drain du noyau exécute **161 instructions ALU entières pour 95 IMAD** et
~32 autres, soit 288 instructions-warp. Ce que cela coûte par SM et par cycle :

| architecture | voies INT32 / SM | cycles ALU | cycles d'émission | goulot | perf / SM·cycle |
|---|---|---|---|---|---|
| Ampere GA10x | 64 | 161/2 = 80,5 | 288/4 = 72 | **ALU** | 1,00 |
| Ada AD10x | 64 | 80,5 | 72 | **ALU** | 1,00 |
| Hopper H100 | 64 | 80,5 | 72 | **ALU** | 1,00 |
| Blackwell GB20x | **128 unifiées** | (161+95)/4 = 64 | 72 | **émission** | **1,12** |

Blackwell fusionne les cœurs INT32 et FP32 : les 128 voies traitent les deux, ce
qui double le débit entier par SM. Le noyau cesse alors d'être limité par l'ALU
et bute sur la limite d'émission de 4 instructions/cycle/SM — d'où 1,12 et non 2.

Conséquence directe et contre-intuitive : **les cartes de datacenter sont un
mauvais choix ici**. H100 et A100 ont les mêmes 64 voies INT32 par SM qu'une
carte grand public, pour 6 à 13 fois le prix horaire. Leurs cœurs tensoriels et
leur HBM — ce qui justifie leur tarif — ne servent strictement à rien à ce noyau.

### 7.3  Coût réel sur vast.ai (relevé septembre 2026)

`h GPU` = 772 / rapport. Le coût total ne dépend **que** des heures-GPU : la
parallélisation n'achète que du temps de calendrier, jamais des euros.

| GPU | SM × GHz | rapport | h GPU | $/h spot | **coût spot** | $/h à la demande | coût à la demande |
|---|---|---|---|---|---|---|---|
| RTX 4070 (référence) | 46 × 2,48 | 1,00 | 772 | — | — | — | — |
| RTX 3090 | 82 × 1,70 | 1,22 | 632 | 0,12 | 76 $ | 0,20 | 126 $ |
| RTX 5070 Ti | 70 × 2,45 | 1,69 | 458 | 0,10 | 46 $ | 0,20 | 92 $ |
| RTX 5080 | 84 × 2,62 | 2,16 | 358 | 0,12 | 43 $ | 0,25 | 90 $ |
| RTX 4090 | 128 × 2,52 | 2,83 | 272 | 0,11 | **30 $** | 0,25 | 68 $ |
| **RTX 5090** | **170 × 2,41** | **4,02** | **192** | **0,15** | **29 $** | **0,32** | **61 $** |
| L40S | 142 × 2,52 | 3,14 | 246 | — | — | 0,55 | 135 $ |
| A100 80 Go | 108 × 1,41 | 1,34 | 577 | — | — | 0,75 | 433 $ |
| H100 SXM | 132 × 1,76 | 2,03 | 379 | — | — | 1,65 | 625 $ |

**La RTX 5090 gagne sur les deux axes à la fois.** Elle coûte le même prix que
la 4090 (29 $ contre 30 $) tout en allant 1,42× plus vite. Une H100 coûterait
**21 fois plus cher** pour aller **deux fois moins vite**.

Temps de calendrier en louant plusieurs 5090 spot — le coût reste ~29 $ :

| 5090 en parallèle | 1 | 4 | **8** | 16 | 32 |
|---|---|---|---|---|---|
| calendrier | 8,0 j | 2,0 j | **24 h** | 12 h | 6 h |

**Huit instances séparées, pas un nœud 8×.** Pour la même durée de 24 h :

| montage | $/GPU/h | **total** |
|---|---|---|
| 8 instances spot séparées | 0,15 | **29 $** |
| 8 instances à la demande | 0,33 | 63 $ |
| un seul nœud 8×5090 (256 vCPU, 504 Go) | 0,60 | **115 $** |

Le nœud multi-GPU coûte **4× le prix** pour exactement le même calcul. La prime
paie un interconnect (NVLink, PCIe entre cartes, RAM partagée) dont ce travail
n'a strictement aucun usage : zéro communication entre workers, 40 octets de
sortie par tâche. Elle n'a de sens que pour la commodité — une seule machine à
configurer, un seul `parts_n31.txt`, `collect.sh` en local. À 86 $ d'écart,
`run_node.sh` rend le montage à 8 instances assez simple pour ne pas la payer.

### 7.4  Plan recommandé

1. **Louer une seule 5090 spot dix minutes (≈ 0,03 $)** et lancer
   `./langford6 -n 31 --bench 64`. Le rapport de 4,02 ci-dessus est un *modèle*
   d'architecture ; ce test le remplace par une *mesure*, sur le même échantillon
   de `vhi` que la 4070. Tout le reste du budget en découle.
2. Choisir le nombre de workers selon le temps voulu, puis lancer
   `./worker.sh <i> <W> 31 4096` sur chacun (4096 tâches ≈ 2,8 min chacune sur
   une 5090 ; avec 8 cartes, 512 tâches par carte).
3. Rassembler avec `./collect.sh 31 4096 parts_*.txt`.

Sur un nœud multi-GPU, `./run_node.sh 31 4096` lance un worker par carte.
**C'est indispensable** : le code prend le device 0 par défaut, donc sans
répartition explicite les 8 processus se battraient pour la carte 0 et les sept
autres resteraient inutilisées — sur un nœud facturé à l'heure, 7/8 du loyer
jeté. `run_node.sh` isole chaque worker par `CUDA_VISIBLE_DEVICES` ;
`--dev <i>` fait la même chose depuis le binaire.

Comparer deux cartes demande le **même nombre d'échantillons** des deux côtés
(la graine est fixe, donc `--bench 64` et `--bench 48` ne tirent pas le même
ensemble). Avec 64 tirages l'erreur-type est de ~8 % ; pour un chiffre absolu
serré, prendre `--bench 256`.

**Le spot est le bon choix ici**, alors qu'il est risqué pour un entraînement :

* chaque tâche est indépendante et **idempotente** — aucune communication entre
  workers, 40 octets de sortie par tâche ;
* une préemption ne coûte que la tâche en cours (~3 min), `worker.sh` saute au
  redémarrage tout ce qui est déjà dans `parts_n31.txt` ;
* `collect.sh` refuse de conclure s'il manque une tâche ou s'il y en a une en
  double, **et** `--merge` vérifie que la somme est divisible par 2^{2n} — un
  test que toute tranche manquante, dupliquée ou corrompue fait échouer (§3).

Attention à l'image : Blackwell (`sm_120`) exige **CUDA ≥ 12.8**. `build.sh` le
détecte et refuse de produire un binaire inutilisable.

### 7.5  Équilibrage des tranches

Le nombre de blocs lancés pour une valeur de `vhi` vaut `NV − vhi` (prédicat de
réflexion), donc la charge décroît linéairement. Mesuré sur les 144 tirages, le
coût brut moyen chute d'un facteur **7** du premier au dernier quartile de `vhi` :

| quartile de `vhi` | [0, ¼) | [¼, ½) | [½, ¾) | [¾, 1) |
|---|---|---|---|---|
| coût brut moyen | 0,560 s | 0,438 s | 0,240 s | 0,080 s |
| divisé par `(NV−vhi)/NV` | 0,637 | 0,690 | 0,627 | 0,671 |

La seconde ligne est plate au bruit près : le modèle est le bon. Un découpage à
`vhi` égaux donnerait donc des tâches allant du simple au septuple. `run_shard.sh`
découpe à **charge** égale en inversant la charge cumulée `v·NV − v²/2` :

    v_i = NV · (1 − √(1 − i/T))

---

## 8. Utilisation

```sh
./build.sh                       # détecte l'architecture (native, sinon sm_XX)

./langford6 -n 24                # run complet (version de référence)
./langford6 -n 31 --bench 64     # calibre CE GPU en ~20 s, projette le total
./langford5 -n 24                # version précédente, pour recoupement croisé
```

**Calcul distribué** (une machine louée = un worker) :

```sh
./worker.sh 0 8 31 4096          # worker 0 sur 8, n=31, 4096 tâches
./worker.sh 1 8 31 4096          # ... sur une autre machine
./collect.sh 31 4096 parts_*.txt # vérifie la complétude puis conclut
```

Rejouer la même commande après une préemption reprend où le worker s'était
arrêté. Les briques de plus bas niveau, si besoin :

```sh
./run_shard.sh 17 4096 31        # une tâche isolée -> ligne PART=...
./langford6 -n 31 --diag         # les orbites fixes (une seule fois)
./langford6 -n 31 --merge <PART...>   # somme + auto-test de divisibilité
```

**Vérification et théorie**

```sh
./oe_check 31                    # vérifie la décomposition de parité
./oe_ref 12 ; ./oe_ref2 12       # références CPU : coordonnées, puis réflexion
./pfaff_test 11                  # Kasteleyn ±1  : impossible
./pfaff_zk 11 32                 # Kasteleyn U(1) : impossible aussi
./dpstates 16                    # états atteignables de la DP (§5.2)
./fiber 12                       # fibres de l'autocorrélation (§3.1)
./zfrac 31 3000000               # fraction de produits non nuls
./estimate 31 600000             # estimation Monte-Carlo
./find_langford 31 5             # 5 suites n=31, vérifiées
```

## 9. Fichiers

**Noyaux de calcul**

* `langford6.cu` — **version de référence** : coordonnées de parité, bitmaps de
  survie, réflexion reparamétrée, produit en complément à deux
* `langford5.cu` — demi-état + déroulage par 8
* `langford4.cu` — saut des termes nuls par compaction warp
* `langford3.cu` — Godfrey + groupe de symétries complet
* `langford2.cu`, `langford.cu` — versions à symétrie ×4, conservées pour
  recoupement croisé
* `langford_ref.c` — référence CPU au plus près de la définition

**Calcul distribué**

* `build.sh` — compilation portable, refuse une architecture que le toolkit
  installé ne connaît pas (Blackwell exige CUDA ≥ 12.8)
* `run_shard.sh` — une tâche, découpée à charge égale (§7.5)
* `worker.sh` — un worker sur une machine louée, reprise après préemption
* `run_node.sh` — un nœud multi-GPU : un worker par carte
* `collect.sh` — vérifie complétude et absence de doublons, puis conclut

**Vérification et théorie**

* `oe_check.c` — décomposition de parité, vérifiée jusqu'à n=31
* `oe_ref.c` / `oe_ref2.c` — références CPU : coordonnées de parité, puis réflexion
* `pfaff_test.c` / `pfaff_zk.c` — réfutation de Kasteleyn (±1 puis U(1))
* `dpstates.cpp` — comptage exact des états atteignables de la DP
* `fiber.c` — fibres de la carte d'autocorrélation
* `parity.c` — l'expérience historique sur la parité des croisements
* `refl_test.c` — validation de σ = canon(reverse) dans les coordonnées de la v3
* `framework.c` — optimalité de Godfrey dans le cadre des poids locaux, §5.4(l)
* `single.c` — la contrainte de couleurs comme équation entière unique, §5.4(j)
* `powersums.c` — combien de sommes de puissances caractérisent l'ensemble plein
* `partial.c` — plus grand ensemble grassmannien admissible, §5.4(i)
* `toeplitz.c` — ordre de la récurrence des hafniens de Toeplitz, §5.4(h)
* `struct.c` — vérification des identités de croisement du §5.4(a) et mesure du
  certificat de décès du §5.4(b)
* `zfrac.c` — fraction de produits non nuls
* `estimate.c` — estimateur de Knuth non biaisé (CPU, OpenMP)
* `find_langford.c` — recherche de solutions, avec vérification indépendante

## Références

**Le problème**

* C. D. Langford, *Problem*, Math. Gazette 42 (1958), 228.
* R. O. Davies, *On Langford's problem II*, Math. Gazette 43 (1959), 253-255.
  (condition d'existence n ≡ 0, 3 mod 4)
* T. Skolem, *On certain distributions of integers in pairs with given
  differences*, Math. Scand. 5 (1957), 57-68.
* [OEIS A014552](https://oeis.org/A014552) — L(2,n) ; [A059106](https://oeis.org/A059106) — variante de Skolem.

**La méthode de comptage**

* M. Godfrey, méthode algébrique (2002). Pas de publication formelle ; décrite
  dans Assarpour–Bar-Noy–Liu §3 et dans D. E. Knuth, *The Art of Computer
  Programming* vol. 4, pré-fascicule 5B, section « Langford pairs ».
* A. Assarpour, A. Bar-Noy, O. Liu, *Counting Skolem Sequences*,
  [arXiv:1507.00315](https://arxiv.org/abs/1507.00315) (2015, rév. 2017).
  L(27) et L(28) ; implémentation CUDA de Godfrey.
* M. Krajecki, C. Jaillet, A. Bui *et al.*, calculs distribués CONFIIT
  (2004-2005) — valeurs jusqu'à n=24.
* D. E. Knuth, *Estimating the efficiency of backtrack programs*,
  Math. Comp. 29 (1975), 121-136. (l'estimateur non biaisé de `estimate.c`)

**Les pistes évaluées**

* P. W. Kasteleyn, *The statistics of dimers on a lattice*, Physica 27 (1961) ;
  *Dimer statistics and phase transitions*, J. Math. Phys. 4 (1963).
  Orientations pfaffiennes.
* L. Lovász, M. D. Plummer, *Matching Theory*, ch. 8 (formulation moderne).
* A. Björklund, *Counting Perfect Matchings as Fast as Ryser*, SODA 2012,
  [arXiv:1107.4466](https://arxiv.org/abs/1107.4466).
* *Counting perfect matchings and Hamiltonian cycles faster*,
  [arXiv:2309.15422](https://arxiv.org/abs/2309.15422) (2023). Hafnien en
  2^{n−Ω(√n)}.
* *A New Direction for Counting Perfect Matchings*,
  [arXiv:1208.2329](https://arxiv.org/abs/1208.2329). Distribution des poids de
  coupe et identité de MacWilliams.
* M. Cygan *et al.*, *On problems as hard as CNF-SAT* / Set Cover Conjecture
  (2016) — la barrière 2^{|U|}.

**Techniques d'implémentation**

* F. Gray, brevet US 2632058 (1953) — code binaire réfléchi.
* R. J. Fisher, H. G. Dietz, *Compiling for SIMD Within A Register*, LCPC 1998.
* S. E. Anderson, *Bit Twiddling Hacks* — détection d'octet nul, étalement de
  bits par multiplication.
* D. E. Knuth, *TAOCP* vol. 2, §4.3.1 — multiplication multi-précision.
* NVIDIA, *CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics* —
  compaction de flux par warp.
