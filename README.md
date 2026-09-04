# Langford L(2,n) — comptage exact sur un seul GPU

Compter **toutes** les suites de Langford d'ordre *n* : arranger 1,1,2,2,…,n,n de
sorte que les deux copies de *k* soient séparées par exactement *k* autres
nombres. Exhiber une solution est trivial ; les compter toutes est un problème
ouvert au-delà de n=28.

Ce dépôt contient une implémentation CUDA de la méthode algébrique de Godfrey,
poussée jusqu'à **5,90× plus vite** que mon point de départ, avec au passage la
fermeture — par preuve ou par mesure — de trois pistes qui étaient jusque-là
seulement « non abouties ».

| | n=24 | n=27 | n=28 | **n=31** |
|---|---|---|---|---|
| point de départ (v3) | 15,8 min | 16,9 h | 2,81 j | 180 j |
| **v6 (ce dépôt)** | **2,7 min** | **2,86 h** | **11,4 h** | **≈ 30,5 j** |

**Ce qui est fait** : le record publié de 2015 — L(28), obtenu sur ~32 GPU en
9 jours — se refait ici en **11,4 heures sur une seule RTX 4070**.
**Ce qui ne l'est pas** : L(2,31) reste inconnu. Ce dépôt n'a pas battu le
record ; il en a divisé le coût par ~600.

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
| v3 — Godfrey + symétrie ×8 | 37,1 | 180,1 j | — |
| v4 — saut des termes nuls | 66,5 | 100,4 j | ×1,79 |
| v5 — demi-état + déroulage | 92,1 | 72,4 j | ×2,49 |
| **v6 — parité + bitmaps de survie** | **437,4\*** | **30,5 j** | **×5,90** |

\* la v6 énumère 2^{2n−2} points nominaux dont la moitié n'est jamais lancée ;
le débit est rapporté à ce total nominal.

### 4.1 v4 — ne pas calculer les produits nuls (×1,79)

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

### 4.2 v5 — ne suivre que la moitié de l'état (×1,39 de plus)

Seuls les écarts pairs décident de la survie ; les 15 impairs ne servent qu'à la
valeur du produit, soit 12,9 % du temps. La boucle chaude ne maintient donc que
les 16 pairs (4 registres au lieu de 8) et reconstruit les impairs depuis `b` au
moment du drain. Plus un déroulage par 8 : j = ctz(t+1) vaut 0,1,0,2,0,1,0,dyn
sur un bloc de 8 pas, donc 7 pas sur 8 adressent la banque de constantes avec un
décalage littéral.

### 4.3 v6 — supprimer la boucle chaude (×2,37 de plus)

C'est le saut structurel. Grâce à la décomposition de parité (§2.2), on
précalcule pour chaque écart *m* et chaque valeur atteignable *v* le **bitmap
des e_lo tels que Q_m(e_lo) = v**. Un thread obtient alors la mortalité de 32
points d'un coup par un OU de 16 mots : le test de survie tombe de ~65
instructions par point à **moins d'une**. Il n'y a plus de code de Gray du tout.

La réflexion reparamétrée (§2.3) rend le prédicat canonique `v ≤ u`, ce qui
permet de **ne pas lancer** les blocs sans travail.

### 4.4 v6 finale — creuser le drain (×1,15 de plus)

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

Drain final : **301 instructions SASS par point survivant**.

### 4.5 Essayé, mesuré, sans gain

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

### 4.6 Trois pièges de mesure, et une leçon d'architecture

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

Et la leçon : **l'empaquetage SWAR n'est pas du gaspillage, c'est de la
compression de registres.** Passer les 15 écarts impairs à l'arbre sous forme
d'entiers économise ~65 instructions sur le papier ; mesuré, c'est **1,85× plus
lent** (376 → 203 Gsums/s), parce que garder 15 entiers vivants fait déborder le
fichier de registres. La version qui marche les multiplie deux à deux (§4.4).

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
| **ce dépôt (v6)** | 1 × RTX 4070 | **11,4 h** | **0,48** |

soit **~600× moins de GPU-jours**. Une part revient au matériel : un GPU Kepler
de 2013 délivre ~1,3·10¹² opérations entières/s contre 7,3·10¹² pour une 4070
(≈ 1,5·10¹³ en comptant le pipe FMA), soit un facteur **6 à 11**. Le reste —
**environ 55 à 100×** — vient de l'algorithme et de l'implémentation : symétrie
d'ordre 8 complète, saut des 87 % de produits nuls, décomposition de parité,
bitmaps de survie, et une arithmétique 160 bits tronquée plutôt que du CRT
modulaire.

**Sur le record lui-même : non, pas encore.** L(2,31) reste inconnu. Ce qui a
changé, c'est son prix :

| | avant | maintenant |
|---|---|---|
| sur une 4070 | 180 jours | **30,5 jours** |
| sur une RTX 4090 | ~67 jours | **~11,3 jours** |
| en location grand public (~0,35 $/h) | ~560 $ | **~95 $** |
| en parallèle | ~26 GPU pendant une semaine | **~4,4 GPU pendant une semaine** |

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
| **v6, mesurée** | **30,5 j** |
| même code à 100 % d'émission (inatteignable) | 20,0 j |
| compte d'instructions à son minimum (~256/survivant), à 100 % | 17,4 j |

43,8 instructions par point canonique, 218,7 Gsums/s canoniques → 9,6·10¹²
instructions/s contre un plafond d'émission de 1,46·10¹³ : **66 % du plafond de
la carte**, et **18 % au-dessus du plancher d'instructions**. Les 30 popcounts
des écarts impairs sont exactement au minimum (8 instructions par écart : 2 SHF,
2 LOP3, 2 POPC, 2 arithmétiques) ; l'arbre de produit est à 1,4× du sien.

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

## 7. Budget pour n=31

Débit effectif mesuré : 437,4 Gsums/s (2^{2n−2} points nominaux).

| n | v3 | v4 | v5 | **v6** |
|---|---|---|---|---|
| 24 | 15,8 min | 8,8 min | 6,4 min | **2,7 min** |
| 27 | 16,9 h | 9,4 h | 6,8 h | **2,86 h** |
| 28 | 2,81 j | 1,57 j | 1,13 j | **11,4 h** |
| **31** | 180 j | 100 j | 72 j | **≈ 30,5 j** |
| 32 | 2,0 ans | 1,1 an | 0,79 an | **≈ 122 j** |

Tout l'état de l'art publié (L(27) + L(28)) se refait en **14 heures** sur cette
carte.

---

## 8. Utilisation

```sh
./build.sh                       # v3/v4/v5/v6 + les outils de vérification

./langford6 -n 24                # run complet (version de référence)
./langford6 -n 31 --from 0 --count 1000    # tranche, pour exécution distribuée
./langford5 -n 24                # version précédente, pour recoupement croisé

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

**Vérification et théorie**

* `oe_check.c` — décomposition de parité, vérifiée jusqu'à n=31
* `oe_ref.c` / `oe_ref2.c` — références CPU : coordonnées de parité, puis réflexion
* `pfaff_test.c` / `pfaff_zk.c` — réfutation de Kasteleyn (±1 puis U(1))
* `dpstates.cpp` — comptage exact des états atteignables de la DP
* `fiber.c` — fibres de la carte d'autocorrélation
* `parity.c` — l'expérience historique sur la parité des croisements
* `refl_test.c` — validation de σ = canon(reverse) dans les coordonnées de la v3
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
