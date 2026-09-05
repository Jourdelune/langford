# Langford L(2,n) — comptage exact sur un seul GPU

Compter **toutes** les suites de Langford d'ordre *n* : arranger 1,1,2,2,…,n,n de
sorte que les deux copies de *k* soient séparées par exactement *k* autres
nombres. Exhiber une solution est trivial ; les compter toutes est un problème
ouvert au-delà de n=28.

Ce dépôt contient une implémentation CUDA de la méthode algébrique de Godfrey,
poussée jusqu'à **6,99× plus vite** que mon point de départ — et **~43× plus
vite que l'état de l'art publié, sur le même matériel** (§4.0) — avec au passage la
fermeture — par preuve ou par mesure — de **quatre** pistes qui étaient
jusque-là seulement « non abouties », dont les tensor cores (§5.2).

| | n=24 | n=27 | n=28 | **n=31** |
|---|---|---|---|---|
| état de l'art publié (2015), sur la même 4070 | ≈ 1,4 h | ≈ 3,6 j | ≈ 14,4 j | **≈ 1 100 j** |
| point de départ (v3) | 15,9 min | 16,9 h | 2,82 j | 180,1 j |
| v6 | 3,0 min | 3,18 h | 12,7 h | ≈ 34,0 j |
| v6.1 | 2,8 min | ≈ 2,95 h | ≈ 11,7 h | ≈ 30,9 j |
| **v7 (ce dépôt)** | **2,8 min** | **2,48 h** ‡ | **≈ 10,6 h** | **≈ 25,8 j** |

‡ n=27 n'est plus une projection : le **total complet a été calculé**, en
2 h 28 min, et rend `L(2,27) = 111 683 611 098 764 903 232` — exactement la
valeur publiée par Assarpour, Bar-Noy & Liu (2015). Voir §3.2(d).

Le gain de la v7 est mesuré **séparément à chaque n** — +1,5 % à n=24, +13,2 %
à n=27, +10,2 % à n=28, **+19,7 % à n=31** — et non interpolé : il dépend
fortement de *n*, et pour une raison qu'on sait nommer (§4.10). Il se trouve
qu'il est le plus grand là où c'est utile, au cas ouvert.

Lignes 2 à 4 : mesurées. Ligne 1 : modélisée — la version publiée n'est pas
réimplémentée ici, son coût est reconstitué à partir de ses deux écarts avec ma
v3 (symétrie ×4 au lieu de ×8, arithmétique modulaire + CRT au lieu d'un bignum
tronqué). Dérivation, incertitude et recoupement au §6, « Le point de départ
vrai ».

**Ce qui est fait** : le record publié de 2015 — L(28), obtenu sur ~32 GPU en
9 jours — se refait ici en **~11,7 heures sur une seule RTX 4070**.
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

> **En une phrase.** L'identité de Godfrey est vérifiée contre une force brute à
> tous les n de 1 à 16, la couverture de l'énumération est prouvée en Lean *et*
> vérifiée exhaustivement sur les 1,07·10⁹ index de n=31, et neuf tranches de
> n=31 — dont les deux régimes extrêmes et le milieu exact — sont identiques au
> bit près à un recalcul par l'**algorithme classique**. Le total complet de
> **n=27 retombe sur la valeur publiée**. Le risque qu'il reste
> une erreur *systématique* est bien en dessous de 1 %. Le risque dominant n'est
> pas là : c'est le **matériel sans ECC** sur 205 heures-GPU, et la seule parade
> complète est de **dupliquer le run** (~23 $ de plus, §3.3bis).

### 3.1 Ce qui est démontré

1. **L'identité de Godfrey.** F homogène de degré 2n en 2n variables ⟹ la somme
   ±1 ne retient que les monômes à exposants tous impairs ; somme des exposants
   = 2n avec 2n exposants impairs ⟹ tous valent 1. Le coefficient multilinéaire
   compte les systèmes ordonnés de cordes couvrant chaque position une fois,
   soit 2·L(2,n). *Vérifiée en plus par calcul exhaustif contre une force brute
   indépendante pour n = 1 à 16 — `./verify 16`, §3.2.*
2. **La décomposition de parité** (§2.2) : calcul d'indices, plus vérification
   numérique exhaustive par échantillonnage jusqu'à n=31.
3. **A_i ≡ i (mod 2)**, donc seuls les écarts pairs peuvent s'annuler : c'est
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
5. **La réduction ×4 n'est valide que pour n ≡ 0 ou 3 (mod 4).** Nier la rangée
   `o` multiplie ∏A_i par (−1)^{MO} et le poids ∏x_k par (−1)^N : la sommande
   n'est invariante que si MO + N est pair — c'est-à-dire exactement aux n où
   une suite de Langford existe. **Ce point manquait, et il coûtait cher** : le
   noyau acceptait `-n 9` et répondait `L(2,9) = 5558`, là où la réponse est 0.
   Un nombre faux, plausible, qu'aucun auto-test n'attrapait. Le binaire refuse
   maintenant ces n ; `./verify 16` montre la divergence à chacun d'eux.
6. **La troncature à 160 bits est exacte.** La somme vraie vaut 2^{2n}·V(n)
   ≈ 2^{145,3} < 2¹⁶⁰ : l'arithmétique modulo 2¹⁶⁰ *est* la réponse exacte, avec
   15 bits de marge.
7. **La couverture de l'énumération, en Lean.** `proof/Langford.lean` prouve —
   sans mathlib, sans `sorry`, `#print axioms` ne rendant que `propext` et
   `Quot.sound` — que tout point a **exactement un** représentant énuméré :
   unicité du translaté épinglé de Klein (`klein_exact`), distinction des quatre
   translatés (`klein_orbit_card_four`, donc le facteur est 4 et pas moins),
   involutivité de σ, trichotomie de la réflexion (`reflexion_exact` : énuméré /
   diagonal / miroir énuméré, les trois s'excluant), et `couverture_exacte`.
   Le théorème vaut pour **un ordre total strict quelconque**, donc il ne dépend
   pas du codage des rangées.

### 3.2 Validation empirique

Quatre vérifications indépendantes, de la plus éloignée du code à la plus
proche. Elles se relancent toutes d'un coup par `./verify_all.sh`.

**(a) L'identité de Godfrey, contre une force brute.** `verify.c` compte trois
fois le même nombre par trois chemins qui ne partagent rien : un retour sur
trace qui énumère les suites (la définition), la somme de Godfrey sur les
2^{2n} points **sans aucune symétrie**, et la même somme réduite par Klein ×4.

| n | 1 – 16, tous |
|---|---|
| force brute = Godfrey complet | **exact partout**, y compris les n où L = 0 |
| Klein ×4 = Godfrey complet | **exact ssi n ≡ 0,3 (mod 4)** — et faux ailleurs |

C'est ce tableau qui a révélé le point 5 du §3.1.

**(b) La couverture, au niveau des bits.** `cover_check.c` réalise le théorème
Lean sur l'indexation *réelle* du noyau — `__brev`, `canon`, le prédicat
`v < u` — et la vérifie **exhaustivement** : involutivité de `rev` et de `f`,
`o(u) = f(u<<1)`, `f(o(u)) = u<<1`, bijection des 2^{N−1} index vers les rangées
épinglées, et le compte 4·(2·C(T,2)+T) = 4^N. **À n=31 cela porte sur les
1 073 741 824 index, un par un.** Plus, pour n ≤ 12, la couverture point par
point de chaque orbite. C'est le pont entre la preuve abstraite et le code.

**(c) Des tranches recalculées depuis la définition.** `slice_ref.c` reprend une
tranche `PART=` et la recalcule sans rien partager avec le noyau : pas de
décomposition de parité, pas de SWAR, pas de PTX, pas de chemin rapide — il
reconstruit la suite X entière et applique A_i = Σ_k x_k x_{k+i}. **28 tranches
comparées, toutes identiques au bit près**, dont cinq **à n=31** avec des
valeurs pleines :

    n=31 vhi=8388607  ->  e4d775be:143d6853:f88bf9c3:5f90e9e5:c8000000
    n=28 vhi=1048575  ->  fffeb84b:1fe955eb:74462f13:059828b1:77400000
    n=27 vhi=524287   ->  fffffd17:c7e8428b:b3dbf5d2:eff1920f:ba400000

C'est ce contrôle-là qui ferme la décomposition de parité, l'arithmétique
160 bits et le chemin rapide **en une fois**, et qui atteint enfin n=27, 28 et
31 — que le §3.3 signalait comme jamais exercés.

`slice_ref` est parallélisé (OpenMP), ce qui met à portée les deux régimes
extrêmes de n=31, et pas seulement la queue bon marché :

| tranche n=31 | points recalculés | régime | durée (18 fils) | |
|---|---|---|---|---|
| `vhi=0` | 1,37·10¹¹ | **shard dégénéré**, 87,9 % de survivants | 15 min | identique |
| `vhi=4194304` | 6,87·10¹⁰ | **milieu exact** de la plage | 7 min 45 | identique |
| `vhi=8000000` … `8388607` | 10⁸ – 10¹⁰ | queue, réflexion élaguante | secondes | identiques |

Les deux régimes extrêmes du §4.7 — celui où presque tout survit et celui où
presque tout est élagué — sont donc vérifiés au bit près **à n=31**, ainsi que
le milieu. Le seul reproche qui subsiste est de porter sur neuf valeurs de
`vhi` sur 8 388 608, pas sur la couverture des cas de figure.

**Valeurs de référence, à recouper sans rien relancer.** Ces sommes partielles
sont produites *deux fois*, par le noyau GPU et par l'algorithme classique, et
coïncident bit à bit. Quiconque implémente Godfrey doit retrouver exactement
ceci — c'est un jeu de tests utilisable indépendamment de ce dépôt :

| n | `vhi` | somme partielle (poids 2, moitié canonique) |
|---|---|---|
| 31 | 0 | `2f608fd7:b7a1f766:d3ce9b3f:99aebb38:08000000` |
| 31 | 4194304 | `9c8f6671:62e2f40e:8c30d79a:eb8cdb2b:48000000` |
| 31 | 8388607 | `e4d775be:143d6853:f88bf9c3:5f90e9e5:c8000000` |
| 28 | 1048575 | `fffeb84b:1fe955eb:74462f13:059828b1:77400000` |
| 27 | 524287 | `fffffd17:c7e8428b:b3dbf5d2:eff1920f:ba400000` |
| 24 | 65535 | `ffffffff:ffd5d6f8:5b23b6fe:7b8715a8:9f300000` |
| 20 | 63 | `00000000:00000000:000001fb:e3903b72:a8a40000` |

Ce tableau n'est pas recopié à la main : il vit dans `refvals.txt`, et
`./check_refs.sh` le recalcule (`./check_refs.sh ref` le refait par l'algorithme
classique). `verify_all.sh` l'inclut, donc une faute de transcription dans le
README ferait échouer la chaîne.

**(d) Les valeurs connues, de bout en bout.** n = 11, 12, 15, 16, 19, 20, 23, 24
reproduites à l'unité près par le binaire courant — et surtout :

> **n = 27 : total complet calculé, 2 h 28 min sur la 4070.**
> `V(27) = 223 367 222 197 529 806 464`, donc
> **`L(2,27) = 111 683 611 098 764 903 232`** — exactement la valeur publiée par
> Assarpour, Bar-Noy & Liu (2015), obtenue par eux sur une grappe de GPU.

C'est le contrôle le plus fort dont on dispose, et il manquait : jusqu'ici aucun
**total** n'avait jamais été calculé au-delà de n=24 avec ce noyau. Il exerce ce
qu'aucune tranche isolée n'exerce — les 524 288 lancements, l'agrégation hôte,
les deux auto-tests de divisibilité sur un vrai total — à un n dont les largeurs
de masque et les comptes d'écarts (14 pairs, 13 impairs) sont proches de ceux de
n=31.

S'y ajoutent les contrôles déjà en place : recoupement v3/v4/v5 identique au bit
près sur n=31 ; `oe_ref`/`oe_ref2` (références CPU en coordonnées de parité) ;
et les deux auto-tests gratuits — divisibilité du total par 2^{2n+1} et de
**chaque tranche** par 2^{⌊(n+1)/2⌋}, ce dernier rejetant une tranche corrompue
sur la machine qui l'a produite. Sur les tranches n=31 rendues : v₂ ≥ 24, soit
8 bits de marge sur les 16 garantis.

### 3.3 Ce que la validation ne couvre toujours pas

**Un run complet à n=28.** Celui de n=27 est fait (§3.2d) et retombe sur la
valeur publiée, ce qui ferme la question de l'agrégation à grand n. Reste n=28,
le second point du record publié : ~10,6 h, soit le trou le moins cher qui
subsiste. Rien n'indique qu'il révélerait quoi que ce soit — n=27 et n=28
partagent le même nombre d'écarts pairs — mais il est bon marché.

**Une erreur d'un seul bit dans les bits hauts d'une tranche.** Les auto-tests
ne contraignent que les bits bas (16 par tranche, 63 sur le total) ; un bit
inversé au-delà passe les deux. `audit.sh` le couvre maintenant **par sondage** :
il rejoue un échantillon aléatoire de tâches et compare au bit près. Un sondage
n'est pas une preuve — rejouer 2 % des tâches attrape une corruption isolée avec
probabilité 2 %.

**Un bug qui donnerait la même réponse fausse dans deux implémentations
indépendantes.** Rien ne l'exclut ; c'est la limite ordinaire de ce genre de
calcul. C'est aussi pourquoi il n'existe **aucun moyen connu de vérifier ce
résultat plus vite que de le recalculer** : pas de certificat succinct, c'est le
corollaire direct du 4ⁿ du §5.

### 3.3bis  Quel risque reste-t-il, chiffré

Deux risques de nature très différente, et c'est le second qui domine.

**Risque systématique — l'algorithme ou le code est faux.** Après le §3.2, ce
qu'il faudrait pour qu'il le soit encore : que l'identité de Godfrey, vérifiée
contre une force brute à tous les n de 1 à 16 ; que la couverture, prouvée en
Lean *et* vérifiée exhaustivement sur les 1,07·10⁹ index de n=31 ; et que neuf
tranches de n=31 identiques au bit près à un recalcul depuis la définition —
dont les deux régimes extrêmes et le milieu — partagent toutes le **même** angle
mort. Je l'estime **bien en dessous de 1 %**. Ce qui subsiste n'est pas une
faiblesse identifiée mais la possibilité générique d'un mode commun.

**Risque transitoire — le matériel se trompe pendant le run.** Une 4090 n'a
**pas d'ECC** : ni sur la VRAM, ni sur les registres, ni sur la mémoire
partagée. Le calcul dure ~205 heures-GPU. Et surtout, nos auto-tests ne
couvrent qu'une partie des positions de bits :

| positions | part | qui les protège |
|---|---|---|
| bits 0–15 | 10,0 % | auto-test par tranche (divisibilité par 2¹⁶) |
| bits 16–62 | 29,4 % | auto-test du total (divisibilité par 2⁶³) |
| **bits 63–139** | **48,1 %** | **rien** |
| bits 140–159 | 12,5 % | l'estimateur de Knuth à ±2 % |

Le calcul : une inversion en position *b* dans une somme partielle décale L de
2^{b−63} ; rapporté à L ≈ 2^{82,25}, l'écart relatif vaut 2^{b−145,25}, et il ne
dépasse les 2 % de l'estimateur qu'à partir de b = 140. **Une inversion d'un bit
au hasard passe donc inaperçue une fois sur deux.**

Reste à savoir combien de telles inversions attendre. Là, honnêtement, je n'ai
pas de chiffre défendable : les taux d'erreurs douces publiés pour de la mémoire
GPU sans ECC varient de plusieurs ordres de grandeur selon l'étude, l'altitude
et le silicium. Mon estimation de praticien, à donner pour ce qu'elle vaut :
**quelques pour cent** de probabilité qu'au moins une corruption non détectée
entache un run de 205 heures-GPU. C'est une opinion, pas une mesure.

**Conclusion opérationnelle : c'est le matériel qu'il faut assurer, pas
l'algorithme.** Et l'assurance est bon marché — le coût total ne dépend que des
heures-GPU (§7.3) :

| stratégie | surcoût | ce que ça attrape |
|---|---|---|
| rien | 0 $ | 52 % des inversions isolées |
| rejouer 2 % des tâches (`SAMPLE`) | ~0,50 $ | + 2 % du reste |
| rejouer 10 % | ~2,30 $ | + 10 % du reste |
| **run entier dupliqué, découpage différent** | **~23 $** | **tout**, y compris une erreur systématique liée au découpage |

Pour un calcul de cette nature, la duplication complète est le bon choix : elle
double une facture de 23 $ et rend le résultat défendable. Deux découpages
différents (par exemple T = 4096 et T = 8192) ne donnent pas les mêmes sommes
partielles ; on compare alors les **totaux**, ce qui teste en prime la logique
de découpage et d'agrégation.

### 3.4 Checklist

**P** = démontré, **L** = prouvé en Lean, **M** = vérifié par calcul,
**✗** = non couvert.

**La méthode donne bien le nombre d'appariements**

| | ce qui doit être vrai | | comment |
|---|---|---|---|
| 1 | l'identité de Godfrey extrait 2·L(2,n) | **P M** | homogénéité §3.1(1) + `./verify 16` : force brute = Godfrey complet, n = 1..16 |
| 2 | la réduction ×4 de Klein est licite | **P M** | MO+N pair ⟺ n ≡ 0,3 (mod 4), §3.1(5) ; `./verify 16` le montre à chaque n ; le binaire refuse les autres |
| 3 | la troncature à 2¹⁶⁰ est exacte | **P** | somme = 2^{145,3}, 15 bits de marge |
| 4 | A_i ≡ i (mod 2) | **P** | §3.1(3) |
| 5 | groupe de symétries d'ordre exactement 8 | **P M** | §3.1(4) + `./fiber 12` |
| 6 | décomposition de parité | **P M** | `./oe_check 31` → 6 200 000 écarts, 0 divergence ; et implicitement par `slice_ref` |
| 7 | l'énumération couvre chaque point **exactement une fois** | **L M** | `proof/Langford.lean` (`couverture_exacte`) + `./cover_check 9 31` exhaustif, **1,07·10⁹ index vérifiés à n=31** |
| 8 | V(n) = 2·L(2,n) est pair | **P** | aucun appariement n'est son propre miroir |
| 9 | toute tranche est divisible par 2^{⌊(n+1)/2⌋} | **P M** | mesuré v₂ ≥ 24 à n=31 |

**Le code calcule bien cette somme**

| | | | |
|---|---|---|---|
| 10 | les valeurs connues sont reproduites | **M** | n = 11..24 **et n=27**, binaire courant |
| 11 | une référence CPU concorde | **M** | `./oe_ref 12`, `./oe_ref2 12` |
| 12 | recoupement entre versions indépendantes | **M** | v3/v4/v5 identiques au bit près sur n=31 |
| 13 | **les chemins de code propres à n=27, 28, 31** | **M** | `./slice_ref` contre le noyau, au bit près : 9 tranches à n=31 dont le **shard dégénéré `vhi=0`** (87,9 % de survivants, 1,37·10¹¹ points) et le **milieu exact** de la plage, plus n=27 et n=28 (§3.2c). *C'était le trou n° 13 ; il est fermé au niveau de la tranche, pas du total.* |
| 14 | une tranche est déterministe (rejeu bit à bit) | **M** | `audit.sh` §5 |
| 15 | un total complet au-delà de n=24 | **M** | **n=27 calculé en entier : `L(2,27) = 111 683 611 098 764 903 232`, la valeur publiée, à l'unité près** (§3.2d). n=28 reste à faire, ~10,6 h |

**Le run de plusieurs semaines n'a pas dérivé**

| | | | |
|---|---|---|---|
| 16 | tranche manquante ou dupliquée | **M** | `audit.sh` §1 |
| 17 | corruption dans les bits bas d'une tranche | **M** | auto-test par tranche, sur la machine productrice |
| 18 | corruption dans les bits 16 à 62 | **M** | auto-test du total, `audit.sh` §2 |
| 19 | corruption dans les bits ≥ 63 d'une tranche | **M (sondage)** | 48 % des positions ne sont protégées par **aucun** test (§3.3bis) ; `audit.sh` §5 rejoue un échantillon, détection = fraction rejouée. **Le seul remède complet est de dupliquer le run (~23 $).** |
| 20 | toutes les tâches ont tourné le **même** code | **M** | `audit.sh` §3 : inventaire des empreintes sha256 portées par chaque ligne |
| 21 | erreur structurelle (symétrie, terme diagonal) | **M** | `./estimate 31` → 5,74·10²⁴ ± 2 % ; toute erreur de ce type décale d'un facteur ≥ 2 |
| 22 | **recalcul complet indépendant** | **✗** | la seule chose qui mérite le nom de preuve ; ~205 h GPU sur une 4090, soit ~23 $ de plus. Recommandé, avec un **découpage différent** (T=4096 puis T=8192) pour tester aussi l'agrégation |

### 3.5 Refaire la vérification soi-même

Toute la chaîne, en une commande :

```sh
./verify_all.sh            # ~3 min
./verify_all.sh full       # ~25 min : ajoute n=16, n=23/24 et plus de tranches
```

Elle enchaîne les sept étapes : identité de Godfrey contre force brute,
couverture exhaustive au niveau des bits jusqu'à n=31, décomposition de parité,
tranches recalculées depuis la définition, valeurs connues de bout en bout,
refus des n illicites, et la preuve Lean. Elle sort non nul si quoi que ce soit
diverge — **à lancer avant toute campagne**.

Contrôles individuels :

```sh
./ladder.sh                     # l'echelle 1..24, n par n, avec sa couverture
./verify 16                     # Godfrey vs force brute, n = 1..16
./cover_check 9 31 12           # couverture exacte, exhaustive, jusqu'a n=31
./slice_ref 31 8388607          # une tranche par l'ALGORITHME CLASSIQUE
./langford6 -n 31 --from 8388607 --count 1 --chunk 1   # la meme, par le GPU
./check_slices.sh 31:8388607 28:1048575 27:524287      # comparaison au bit pres

# les deux regimes extremes de n=31, plus couteux mais a portee :
OMP_NUM_THREADS=18 ./slice_ref 31 4194304   # milieu exact   ~8 min sur 18 fils
OMP_NUM_THREADS=18 ./slice_ref 31 0         # shard degenere ~15 min
(cd proof && lean Langford.lean)                       # la preuve
```

### 3.6 Faire relire le résultat par quelqu'un d'autre

Chaque tâche distribuée écrit désormais sa **provenance** à la suite de sa somme
partielle : empreinte sha256 du binaire, carte, pilote, plage de `vhi`, durée,
horodatage UTC. Les deux premiers champs ne bougent pas, donc `collect.sh`
continue de fonctionner tel quel.

```
#417 00000000:0000102f:f39eee83:d063b35b:56800000 task:417/4096 n:31 \
     vhi:1904..1912 sha:0cfba5123669308d gpu:NVIDIA_GeForce_RTX_4090 \
     drv:570.86.10 sec:181 utc:2026-09-05T09:12:44Z
```

`./audit.sh 31 4096` produit alors un dossier d'audit en cinq points :
complétude et unicité des tâches, auto-test arithmétique sur le total,
**inventaire des binaires** (toutes les tâches doivent porter la même empreinte),
inventaire des cartes et pilotes, et **recalcul redondant d'un échantillon
aléatoire** comparé au bit près. `SAMPLE=50 ./audit.sh 31 4096` rejoue 50 tâches.

Ce qu'il faut publier pour qu'un tiers puisse conclure sans refaire le calcul :
le fichier `parts_n31.txt` complet (8 193 lignes avec provenance), la sortie de
`verify_all.sh`, celle d'`audit.sh`, et l'empreinte du binaire avec le commit
correspondant. N'importe qui peut alors refaire l'addition, rejouer les
auto-tests, et recalculer les tranches de son choix.

## 4. Les optimisations, dans l'ordre, avec les gains mesurés

Tous les débits sont mesurés GPU au repos, moyennés sur des shards répartis
(voir §4.6 : mesurer au mauvais endroit m'a coûté plusieurs heures).

### 4.0  La chaîne complète, de 2015 à aujourd'hui

Toutes les lignes donnent le coût de **n=31 sur la même RTX 4070**, ce qui rend
les gains comparables entre eux. Le facteur de chaque ligne est son gain sur la
ligne précédente ; le produit de la colonne vaut le total.

| # | ce qui change | n=31 | gain | origine | facteur établi par |
|---|---|---|---|---|---|
| | **Godfrey nu (2002)** — aucune symétrie, arithmétique modulaire + CRT | ≈ 4 500 j | — | **littérature, Langford** — Godfrey 2002 | |
| 1 | **symétrie d'ordre 4** — épingler 2 coordonnées → **l'état de l'art publié** | **≈ 1 100 j** | **×4,1** | **littérature, Langford** — Assarpour, Bar-Noy & Liu 2015 §4 | comptage de points, 2⁶² → 2⁶⁰ |
| 2 | **bignum 160 bits tronqué** au lieu de modulaire + CRT (v1/v2) | 360 j | ×3,06 | **hors Langford** — arithmétique tronquée en complément à deux, Knuth *TAOCP* II §4.3.1 ; le choix et sa mesure sont à moi | **modèle** arithmétique (§6) |
| 3 | **symétrie d'ordre 8** — la réflexion en plus du groupe de Klein (v3) | 180,1 j | ×2,00 | **mixte** — les symétries sont listées par Assarpour *et al.*, qui n'en tirent qu'un facteur 4 ; le facteur 8 complet est à moi | comptage de points, 2⁶⁰ → 2⁵⁹ |
| 4 | **ne pas calculer les produits nuls** — 87 % le sont ; compaction en file par warp (v4, §4.1) | 101,7 j | ×1,77 | **mixte** — la compaction par warp est une technique NVIDIA classique ; l'appliquer ici lève une objection SIMT qui faisait rejeter l'idée (§6.bis 5) | mesure appariée |
| 5 | **demi-état + déroulage par 8** — seuls les écarts pairs décident de la survie (v5, §4.2) | 75,0 j | ×1,36 | **à moi** — conséquence d'ingénierie de A_i ≡ i (mod 2), qui est classique | mesure appariée |
| 6 | **coordonnées de parité + bitmaps de survie** — la boucle chaude disparaît (v6, §4.3) | 34,0 j | ×2,21 | **à moi** — la décomposition deux rangées est connue pour l'argument n ≡ 0,3 (mod 4), l'exploiter comme *séparation de variables* ne l'est pas (§6.bis 1) | mesure appariée |
| 7 | **chaînes de retenue PTX + extraction `prmt`** (v6.1, §4.9) | 30,9 j | ×1,10 | **hors Langford** — idiomes standard (CGBN, ISA PTX) ; l'apport est de les avoir mesurés ici et vérifié que ptxas ne les trouve pas | mesure appariée |
| 8 | **écarts impairs tabulés + chemin rapide du produit + `LDS.64`** (v7, §4.10) | **25,8 j** | **×1,20** | **à moi** — la tabulation est une technique banale ; l'axe de décomposition (`threadIdx` ne pilote que 8 bits de `o`, via `__brev`) ne l'est pas | mesure appariée |

> **De l'état de l'art publié (2015) à la v7, sur le même matériel : ×42,6.**
> Depuis Godfrey nu : ×174. Depuis mon point de départ mesurable (v3) : ×6,99.

**Lecture de la colonne « origine ».** *Littérature, Langford* : publié, et déjà
appliqué à ce problème — je ne fais que l'implémenter. *Hors Langford* :
technique connue ailleurs (arithmétique multi-précision, idiomes PTX), dont
l'apport ici est le choix et la mesure, pas l'invention. *Mixte* : le mécanisme
existe, son application à ce problème est à moi. *À moi* : je ne l'ai vue nulle
part, ce qui ne prouve pas qu'elle n'y est pas. Le §6.bis détaille chaque
attribution, y compris ce que je considère comme du folklore.

**Ce que ça donne comme partage.** Le ×4,1 qui mène de Godfrey nu à l'état de
l'art publié vient entièrement de la littérature Langford : je ne fais que
l'implémenter. Tout ce qui suit — le ×42,6 — a été **ajouté ici**, ce qui n'est
pas la même chose qu'inventé ici : les lignes 2, 3, 4 et 7 reposent sur des
techniques connues ailleurs, elles n'avaient simplement pas été appliquées à ce
problème. En décomposant :

| | facteur |
|---|---|
| une seule idée vraiment structurelle — la séparation de variables du §2.2 (ligne 6) | ×2,21 |
| tout le reste : arithmétique, symétrie complète, compaction, PTX, tabulation (lignes 2, 3, 4, 5, 7, 8) | ×19,4 |
| **total ajouté ici** | **×42,6** |

Autrement dit, **l'essentiel du gain est de l'ingénierie**, pas une idée. C'est
cohérent avec le §6 : l'exposant n'a pas bougé depuis 2002, et ce dépôt ne
prétend pas le contraire.

Trois précautions sur ce tableau.

* **Une seule ligne est vraiment un modèle : la ligne 2.** La version publiée
  n'est pas réimplémentée ici ; son coût est reconstitué à partir de ses deux
  écarts avec ma v3 (symétrie ×4 au lieu de ×8, arithmétique modulaire + CRT au
  lieu d'un bignum tronqué). Les facteurs ×4,1 et ×2,00 des lignes 1 et 3, eux,
  sont exacts *par construction* : ils ne font que compter les points énumérés.
  L'incertitude porte donc sur le ×3,06, d'où la fourchette 740 à 1 380 jours
  pour la ligne 1 et un total de **×29 à ×54** (dérivation au §6).
* **Les lignes 4 à 8 sont des rapports mesurés**, chacun GPU au repos et
  apparié sur le même échantillon de `vhi` — c'est le rapport qui est fiable,
  pas les absolus, qui portent ±8 % (§7.1).
* **Deux comptabilités coexistent.** Ce tableau est sur la base des débits
  nominaux en Gsums/s. La mesure directe de bout en bout de la v7 donne 24,4 j
  plutôt que 25,8, soit ×45 depuis 2015 — c'est le chiffre du §6. L'écart de
  6 % est dans les ±8 % ci-dessus.

### 4.0bis  Le détail par version

| version | Gsums/s | n=31 | gain cumulé depuis v3 |
|---|---|---|---|
| v3 — Godfrey + symétrie ×8 | 37,0 | 180,1 j | — |
| v4 — saut des termes nuls | 65,6 | 101,7 j | ×1,77 |
| v5 — demi-état + déroulage | 89,0 | 75,0 j | ×2,40 |
| v6 — parité + bitmaps de survie | 392,7\* | 34,0 j | ×5,30 |
| v6.1 — chaînes de retenue + PRMT | 432,5\* | 30,9 j | ×5,84 |
| **v7 — écarts impairs tabulés** | **517,7\*** | **25,8 j** | **×6,99** |

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
mesurée sur la v6 : produit 64 %, écarts impairs 55 %, base 30 % — les parts se
recouvrent, parce qu'elles viennent de suppressions qui masquaient aussi de la
latence. **Le profil de la v6.1, refait avec un protocole qui ne se recouvre
pas, est au §4.11** ; il déplace le problème (la « base » est dans la compaction
et le balayage, pas dans les tables).

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

Drain final : **287 instructions SASS par point survivant** — ramené à **251**
par la v6.1 (§4.9).

*La v7 (§4.10) généralise cette section aux **trente** popcounts, et pas
seulement à huit : ce n'est pas `e_lo` qu'il faut regarder, mais le fait que
`threadIdx` ne pilote que huit bits de `o`.*

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
* **Instructions vidéo PTX** (`vmad.s32.s32.s32 d, a.b0, a.b1, 0`), qui
  calculeraient le produit de deux octets signés extraits *en une instruction*.
  Sur Ada elles ne sont plus câblées : ptxas les développe en **deux PRMT par
  opérande** plus l'IMAD, soit 9 instructions là où le §4.9 en met 3. Vérifié
  au désassemblage, jamais mesuré — inutile.
* **Parité du signe en un seul popcount.** `popc(a)+popc(b) ≡ popc(a^b) (mod 2)`,
  donc `sgn` peut se calculer avec un POPC au lieu de deux. Exact, mais ptxas
  rend le même compte d'instructions (2 POPC + IADD3 → 1 POPC + LOP3 + 2 NOP de
  bourrage) : **0,0 % à trois tours de mesure**. Non retenu — la ligne d'origine
  est validée, on ne la touche pas pour rien.
* **Vectoriser le balayage** (`LDS.128`) et **supprimer les lectures partagées
  du drain** : 5 % et 4,5 % plus lent respectivement. Détail et raison au §4.10.
* **Constantes d'écart pré-additionnées.** Les huit R(m) du §4.5 étaient
  rangées en int16 et le drain leur ajoutait `L2`, une constante de
  compilation, par un IADD3 par écart. En rangeant plutôt `R(m) + L2`, qui tient
  sur un **octet** signé (dans [−1, 45] à n=31), on retire les 8 IADD3 et deux
  LDS, et l'extraction reste à une instruction (`prmt` au lieu de
  `LEA.HI.SX32`). Compté au désassemblage : **1 520 → 1 504 instructions**.
  Mesuré : **3,5 % plus lent**, reproductible sur deux tours (678,9 contre
  656,1 h). Seize instructions de moins et une perte nette — le §4.8 conclut
  que « seul le compte d'instructions compte », et c'est le contre-exemple : le
  compte ne suffit pas quand il déplace la pression sur un port (ici 10 `prmt`
  de plus) ou change le stride d'un tableau partagé. Non retenu ; la v7 rend le
  point sans objet, puisqu'elle supprime ces constantes.
* **Rebalayage de K après la v6.1.** Le drain ayant maigri de 12,5 %, l'optimum
  aurait pu se déplacer. Non : K=8 mesure 796 h (registres 64 → 71, et 34,8 Ko
  de mémoire partagée qui font tomber l'occupancy), K=6 mesure ~1 570 h. **K=7
  reste l'optimum**, avec la même marge qu'avant.

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

(Le profil complet, refait sur la v6.1 avec deux protocoles indépendants, est
au §4.11.)

**Conclusion.** Le noyau est à ~80 % du plafond d'émission que lui impose son
propre mélange d'instructions (une instruction ALU occupe son pipe deux cycles,
une FMA un seul). Il n'est donc ni à optimiser par l'occupancy, ni par les
pipes, ni par la mémoire : **seul le nombre d'instructions compte encore**. Et
il est à 15 % de son plancher, avec les 22 popcounts des écarts impairs
exactement au minimum et l'arbre de produit à ~1,4× du sien. *(Ce dernier écart
est celui que la v6.1 est allée chercher, et il n'existe plus : l'arbre est
désormais au plancher du schoolbook, §4.9. Les décomptes 161 ALU / 95 IMAD de
cette section décrivent la v6 ; la v6.1 donne 95 IADD3 pour 455 IMAD.)*

**Contrôle du modèle (v6.1).** Cette conclusion se teste : si seul le compte
d'instructions compte, alors remplacer les 23 POPC du drain par un `& 31` — même
graphe de dépendances, un LOP3 au lieu d'un POPC — doit rapporter *exactement*
ce que vaut la différence de compte, ni plus. Mesure : la sonde retire 12
instructions nettes par survivant sur ~250 et gagne **4,9 %** (deux tours,
617,8 h contre 648,0 h). Le modèle prédit 4,8 %. Le POPC n'a donc **pas** de
pénalité de débit cachée sur Ada, et toute idée qui échangerait des popcounts
contre des lectures en mémoire partagée est perdante d'avance — ce qui ferme la
généralisation du §4.5 à tous les écarts impairs.

### 4.9  v6.1 — ce que le C ne sait pas dire (×1,101)

Le §4.8 conclut que seul le compte d'instructions compte encore. Deux endroits
du drain en portaient plus que nécessaire, et pour la même raison : nvcc traduit
fidèlement ce que le C exprime, et le C n'a de mots ni pour une **chaîne de
retenue**, ni pour une **extraction signée d'octet**. Aucun des deux changements
ne touche à l'algorithme ; les deux sont des réécritures locales en PTX.

**1. L'arbre de produit, en chaînes de retenue (+6,06 %).**

L'étage final formait ses neuf produits partiels en `uint64_t` puis recombinait
les moitiés à la main. ptxas n'a alors pas le choix : chaque produit occupe une
paire de registres, et les retenues se propagent par des IADD3/IADD3.X séparés —
les ~21 instructions de propagation annoncées au §5.3. Les formes
`mad{c}.{lo,hi}.cc` du PTX **absorbent l'addition et la retenue dans le
multiplieur** : une IMAD par produit partiel, zéro addition explicite.

    ligne c_j :  passe sur les moities BASSES  (une chaine de retenue)
                 passe sur les moities HAUTES  (l'autre)

Deux passes par ligne parce qu'il n'y a **qu'un seul** drapeau de retenue : les
`lo` et les `hi` d'une même ligne ne peuvent pas s'entrelacer. Les retenues qui
sortent du limbe 4 sont jetées — la troncature à 160 bits est gratuite. L'étage
96×96 tombe de ~30 à **17 instructions**, l'étage 64×64 de ~9 à **7**.

Deux pièges, tous deux silencieux :

* le drapeau de retenue n'est pas modélisé par le compilateur, donc **une chaîne
  doit tenir dans un seul bloc `asm`** — sinon ptxas peut y glisser une
  instruction qui l'écrase ;
* les sorties doivent être en `"=&r"` (earlyclobber). Sans le `&`, le
  répartiteur a le droit d'aliaser une sortie écrite tôt avec une entrée relue
  plus tard dans la chaîne, et le résultat est faux une fois sur ~2³².

Effet mesuré sur le mélange : **IADD3 149 → 95**, IMAD 429 → 455. C'est aussi
l'échange que l'hypothèse 1 du §4.8 avait tenté et raté (`__umulhi` pour les
décalages, 4 % plus lent) — la différence est que celui-ci *retire* des
instructions au lieu d'en déplacer.

**2. L'extraction des octets SWAR, en une instruction (+3,9 %).**

Le drain lit 16 valeurs d'écart pair empaquetées en quatre mots SWAR, et doit
étendre chaque octet en entier 32 bits **signé**. `prmt.b32` fait exactement
cela : quand le bit 3 d'un sélecteur est mis, l'octet de sortie vaut 0x00 ou
0xFF selon le **signe** de l'octet choisi. Un sélecteur `(8|j, 8|j, 8|j, j)`
étend donc l'octet *j*, seul, en une instruction.

nvcc ne connaît ce motif que pour l'octet 0 ; pour les octets 1 et 2 il émet
décalage + xor + décalage arithmétique. Compté au désassemblage : **9
instructions par mot SWAR au lieu de 5**, sur 4 mots et pour chaque survivant.
Les quatre sélecteurs sont 0x8880, 0x9991, 0xAAA2, 0xBBB3.

    PRMT 16 → 40      SHF 194 → 170      LOP3 252 → 236

**Bilan.** `oe_kernel<31,7>` passe de **1 592 à 1 520 instructions SASS**. Le
drain y figure deux fois (paquet plein et queue), donc **−36 instructions par
survivant : 287 → 251**, soit −12,5 %.

| | n=24 (bout en bout) | n=31 (`--bench`) |
|---|---|---|
| v6 | 197,11 s | 714,6 h / 720,2 h |
| **v6.1** | **185,78 s** | **648,9 h / 654,0 h** |
| gain | **+6,10 %** | **+10,13 %** |

Les deux gains sont mesurés séparément (+6,06 % pour le premier seul,
+10,23 % pour les deux) ; le total ci-dessus est remesuré d'un bloc à la fin,
d'où le +10,13 % plutôt que le produit exact des deux.

Protocole : GPU au repos, mesures **appariées et alternées** base/v6.1 pour
absorber la dérive d'horloge, deux échantillons de `vhi` indépendants
(`--bench 64` et `--bench 96`), deux tours chacun. Les quatre tours donnent
+10,19 / +10,06 / +10,09 / +10,18 % — **reproductible à 0,07 %**, alors que
chaque valeur absolue porte ±8 % d'erreur d'échantillonnage. C'est le rapport
qui est mesuré ici, pas les absolus (§4.7, piège n° 4).

Le gain dépend de *n* parce que le drain n'est pas la même fraction du travail :
à n=24 il y a 12 écarts impairs et 12 pairs contre 15 et 16 à n=31, et les
phases de construction des tables, elles, ne changent pas.

**Ce que ça ne change pas.** Trois contrôles, et il faut dire ce que chacun
couvre exactement :

1. **Les huit valeurs connues** (n = 11, 12, 15, 16, 19, 20, 23, 24) sont
   reproduites à l'unité près par le run complet. Cela exerce tout le chemin,
   mais pas aux valeurs de n=31.
2. **L'arbre de produit ne dépend pas de n.** `prod160m` n'est ni templaté ni
   paramétré par N : son équivalence bit-à-bit avec la version d'origine a été
   vérifiée sur 2²⁰ tirages couvrant tout le domaine réel des A_i (uniformes
   dans [−61, 61], plus un huitième de cas extrêmes à ±61, où les produits
   partiels sont les plus grands). **0 divergence.** Ce contrôle-là vaut donc
   *aussi* pour n=31.
3. **L'extraction PRMT, elle, est dans un noyau dépendant de n.** Le noyau
   diagonal sort des sommes **identiques au bit près** à n = 11, 16, 20, 24,
   **27, 28 et 31** : les largeurs de masque et le nombre d'écarts propres à
   n=31 — que le §3.3 signale comme non couverts par le reste de la validation —
   sont donc exercés pour ce changement-ci.

4. **Et le contrôle direct**, le seul qui porte sur le noyau principal *à*
   n=31 : cinq tranches réelles recalculées par les deux binaires, réparties sur
   toute la plage de `vhi` (0 — le shard dégénéré du §4.7 —, 1 000, 4 698 750,
   8 000 000, et la queue où la réflexion élague tout). **Sommes partielles
   identiques au bit près.**

Conséquence pratique : les lignes `PART=` déjà produites par la v6 restent
valides et s'agrègent sans réserve avec celles de la v6.1. La géométrie des
tranches ne change pas non plus (K reste à 7), donc une campagne interrompue
reprend simplement plus vite.

Ce que ça ne couvre toujours pas : un run complet de `oe_kernel` à n=27 ou 28.
Le trou du §3.3 reste ouvert, ni plus ni moins qu'avant.

**Un bug latent trouvé au passage.** `sPw[256][9]` codait en dur le nombre de
mots réservés aux R(m) constants du §4.5. Neuf est le bon compte pour n=31 et
K=7 — et pour eux seuls : à K=6 il en faut dix, et le dixième mot débordait dans
la file des survivants du bloc, d'où un accès illégal. Le compte est maintenant
dérivé de (N, K). C'est neutre à K=7, à l'octet et à l'instruction près.

### 4.10  v7 — les écarts impairs deviennent, eux aussi, une somme de deux tables (×1,197)

Le §4.5 avait remarqué qu'*une* des deux corrélations de l'écart impair *m* ne
dépend pas de `e_lo` dès que m ≥ K+1, et en avait tiré 8 des 30 popcounts du
drain. Le §4.8 avait ensuite conclu qu'il n'y avait plus rien à prendre de ce
côté : sa sonde « POPC → `& 31` » montre que le POPC n'a aucune pénalité de
débit cachée sur Ada, donc **échanger un popcount contre une lecture en mémoire
partagée est perdant d'avance**.

Cette conclusion est juste, et elle ne ferme pas ce qui suit — parce que la v7
n'échange pas les popcounts contre des lectures. Elle les **sort du drain** vers
un endroit où ils sont amortis sur ~16,5 survivants.

**Ce que `__brev` donnait gratuitement depuis la v6.** Le noyau construit `o`
par `rvF = __brev(u<<1) >> (32−N)` avec `u = blockIdx·256 + threadIdx`. Les bits
de poids **faible** de `u` — donc exactement `threadIdx` — atterrissent dans les
bits de poids **fort** de `rvF`. D'où un fait que je n'avais jamais exploité :

> à l'intérieur d'un bloc, `o` ne varie que sur les **huit** bits [N−9, N−2].
> Tout le reste de `o` est constant sur le bloc.

(Le bit N−1 vaut 0 avant canonicalisation ; le drapeau de canonicalisation
`rvF & 1` est le bit N−2 de `u`, donc un bit de bloc lui aussi.)

**La décomposition.** Un écart impair est une somme de produits O_a·E_c. On
classe les couples (a, c) selon que *a* tombe dans la fenêtre du thread et *c*
dans `e_lo` :

| a ∈ fenêtre thread | c ∈ `e_lo` | dépend de | rangé où |
|---|---|---|---|
| non | non | (bloc, e_hi) | mot SWAR de `sPw` |
| **oui** | non | (thread, e_hi) | mot SWAR de `sPw` |
| non | **oui** | (bloc, `e_lo`) | **table `sF`** |
| **oui** | **oui** | les deux | **3 termes** à n=31/K=7 |

Les deux premières lignes se calculent ensemble, et sans effort : leur somme est
simplement A_{2m+1}(o, `e_lo` = 0). La troisième ne dépend pas de `e_hi`, donc
`sF` se construit **une fois par bloc**, pas une fois par `vhi`. La quatrième se
réduit à trois produits de bits, traités au vol.

Les 15 écarts impairs sortent donc, comme les 16 pairs, d'une simple **addition
SWAR de deux mots lus en mémoire partagée** :

    ecart impair = sF[e_lo] + sPw[thread]     (octets, biais 64 de chaque cote)

Le 16ᵉ octet porte le signe : `sPw` y met popc(o)+popc(e_hi), `sF` y met
popc(`e_lo`), et la parité de leur somme décide. Le drain ne fait **plus aucun
popcount**.

`check_decomp.c` vérifie l'identité exactement, terme à terme, report des termes
croisés compris : **0 divergence** sur 3,0·10⁶ écarts tirés à n=31, et le même
contrôle passe pour n = 11 à 28. Il imprime aussi les termes croisés : à n=31 et
K=7 il y en a **trois**, sur les écarts m=14 et m=15.

**Deux ajouts qui ne touchent pas non plus au résultat.**

* **Chemin rapide dans l'arbre de produit.** Le pire cas exige 160 bits — un
  terme peut valoir 2¹⁶⁹ — mais le terme *typique* vaut 2⁶⁴ : |A_i| ~ √(2n−i),
  donc E[log₂ ∏|A_i|] = 64,5 avec un écart-type de 8,9. Quand les quatre
  facteurs 64 bits de l'étage médian tiennent sur un `int32` — test exact, deux
  instructions chacun — le produit tient sur 126 bits et **une seule**
  multiplication 64×64 remplace deux `mul64_96` *plus* `mul96_160`. Le chemin
  lent reste là pour le reste, donc le résultat est inchangé au bit près.
* **Strides de 6 et 10 mots** pour `sT`, `sF` et `sPw` : multiples de 8 octets
  (donc `LDS.64` légal) et de moitié **impaire** (3 et 5), donc les paires de
  bancs restent distinctes sur seize voies. Les huit mots SWAR du drain se
  lisent en quatre `LDS.64` : 18 LDS → 10.

**Mesuré.**

| | v6.1 | **v7** |
|---|---|---|
| corps de la boucle interne (SASS) | 288 | **225** puis 262† |
| dont POPC | 26 | **0** |
| registres / spill | 64 / 0 | 80 / 0 |
| blocs par SM | 3 | 3 |
| n=31, `--bench 64` | 657,3 / 663,1 h | **551,6 / 551,5 h** |

† la décomposition seule tombe à 225 ; le chemin rapide *ajoute* du code
statique — les deux branches sont dans la boucle — tout en retirant des
instructions *exécutées*. Rappel utile que le compte statique ne décide de rien
ici : le §4.6 contient d'ailleurs une réécriture qui retire 16 instructions du
noyau et mesure **3,5 % plus lent**.

Les trois étages, mesurés séparément et appariés sur le même échantillon de
`vhi` :

| | h GPU | gain cumulé |
|---|---|---|
| v6.1 | 657,3 / 663,1 | — |
| + tables d'écarts impairs | 600,2 / 600,6 | +9,9 % |
| + chemin rapide du produit | 562,5 / 562,5 | +17,4 % |
| **+ lectures `LDS.64`** | **551,6 / 551,5** | **+19,7 %** |

La v7 est aussi bien plus **stable** d'un tirage à l'autre : 551,56 puis
551,46 h sur deux tours (0,02 %), là où la v6.1 donne 657,3 puis 663,1 (0,9 %).
Cohérent avec la disparition des popcounts, dont le coût dépendait du motif de
bits.

**Le gain dépend beaucoup de *n*, et on sait pourquoi.** Il vaut +19,7 % à
n=31 mais **+1,5 %** à n=24. La cause est la dernière ligne du tableau de
décomposition : le nombre de termes croisés, ceux que la table ne peut pas
porter et que le drain doit refaire à la main. Ce nombre est piloté par
l'écart entre la fenêtre du thread — les bits [N−9, N−2] — et `e_lo`, les bits
1..K. Plus *n* est grand, plus les deux fenêtres s'éloignent, et moins il reste
de couples (a, c) où les deux tombent dedans. `check_decomp.c` les compte :

| n | 20 | 23 | 24 | 27 | 28 | **31** |
|---|---|---|---|---|---|---|
| termes croisés restants | 35 | 21 | 21 | 10 | 10 | **3** |
| gain mesuré | — | — | +1,5 % | +13,2 % | +10,2 % | **+19,7 %** |

À n=24 les deux fenêtres se touchent presque et vingt et un produits de bits
reviennent dans le drain ; à n=31 il n'en reste que trois. **L'optimisation est
donc la plus efficace exactement là où elle sert** — au cas ouvert. C'est aussi
pourquoi je ne l'interpole pas : les mesures de n=27 et n=28 (`--bench 32`,
donc plus bruitées que celles de n=31) sortent d'ailleurs dans le mauvais ordre
l'une par rapport à l'autre, ce qui donne l'échelle du bruit sur ces deux
points-là.

**Une question fermée au passage.** Les 80 registres de la v7 tiennent tout
juste 3 blocs par SM. J'ai donc mesuré la variante à 2 blocs (`-DLF6_MINBLK=2`,
110 registres, ptxas cesse de rematérialiser les invariants de boucle) :
**702,8 h**, soit 27 % plus lent. Le §4.8 concluait l'occupancy indifférente
*entre 16 et 40 warps* ; sur ce noyau-ci, à 16 warps, elle ne l'est plus. Trois
blocs par SM est le bon point, et c'est lui qui fixe le budget de registres.

**Ce que ça ne change pas.** Les huit valeurs connues (n = 11, 12, 15, 16, 19,
20, 23, 24) sont reproduites à l'unité près. Le noyau diagonal sort la même
somme au bit près à n=31. Et le contrôle qui porte vraiment, celui du §4.9 :
**cinq tranches réelles de n=31** — `vhi` = 0 (le shard dégénéré du §4.7), 1 000,
4 698 750, 8 000 000, et la queue où la réflexion élague tout — recalculées par
les deux binaires donnent des sommes partielles **identiques au bit près**. Les
lignes `PART=` déjà produites par la v6/v6.1 restent donc valides et s'agrègent
sans réserve.

Ce que ça ne couvre toujours pas : un run complet de `oe_kernel` à n=27 ou 28.
Le trou du §3.3 reste ouvert, ni plus ni moins qu'avant.

**Ce qui n'a pas été refait, et devrait l'être.** Trois choses, dont deux
pourraient encore rapporter :

* **Le balayage de K.** K=7 était l'optimum de la v6.1, et l'arbitrage a changé
  du tout au tout : les 30 popcounts par (thread, vhi) coûtent maintenant plus
  cher relativement, et K=8 les amortirait sur deux fois plus de `e_lo`. Mais
  K=8 double `sB` et ferait tomber l'occupancy à 2 blocs par SM — précisément
  ce que la mesure ci-dessus condamne. À vérifier plutôt qu'à supposer.
* **Le profil du §4.11**, refait sur la v6.1, ne décrit plus le drain de la v7 :
  la compaction et le balayage y pèsent forcément plus lourd maintenant que le
  drain a fondu.
* **Le rapport 5090/4070 (§7.2, §7.3)**, mesuré sur la v6.1. Le mélange
  d'instructions a changé ; le rapport aussi, peut-être. *(Le rapport 4090,
  lui, est maintenant mesuré sur la v7 : 2,66, §7.3.)*

### 4.11  Où va le temps (profil mesuré sur la v6.1)

Le §4.4 datait de la v6 et donnait des parts qui se recouvraient (« produit
64 %, écarts impairs 55 %, base 30 % »), parce qu'elles venaient de suppressions
qui masquaient aussi de la latence. Le profil ci-dessous est refait sur la v6.1
avec deux protocoles distincts, et il ne se recouvre pas :

* **suppression** pour les deux gros blocs du drain (on les remplace par un
  calcul trivial gardant les mêmes dépendances) ;
* **duplication** pour tout le reste — on fait le travail *deux fois*, ce qui
  donne le coût marginal d'une passe sans toucher ni aux registres ni au
  résultat. Les sondes de duplication produisent d'ailleurs toujours la bonne
  valeur de L(2,20), ce qui les valide.

| composant | part du temps | protocole |
|---|---|---|
| arbre de produit 160 bits | **22 %** | suppression |
| écarts impairs (22 popcounts) | **24 %** | suppression |
| compaction des survivants (file par warp) | **~11 %** | duplication |
| balayage des bitmaps de survie (phase 3) | **~9 %** | duplication |
| construction des tables (phases 1-2) | **~5 %** | duplication |
| reste : lectures partagées du drain, ADD160, R(m), barrières, transfert hôte | ~29 % | par différence |

Deux enseignements.

**La « base » n'est pas 30 %, elle est ~25 % — mais elle n'est pas dans les
tables.** Ce sont la compaction et le balayage qui la portent, pas la
construction des bitmaps (5 %). Et les deux résistent :

* le balayage fait déjà 16 LDS + **8** LOP3 par mot, pas 15 : ptxas émet des
  `LOP3` à **trois** entrées (motif `0xfe`), donc réduire 16 valeurs lui coûte
  déjà ⌈15/2⌉ instructions. Il n'y a rien à gagner là où je croyais ;
* la compaction tourne au **maximum** du nombre de survivants sur les 32 voies,
  pas à leur moyenne : 8,3 tours contre 4,1 utiles, soit un facteur 2 structurel.
  Fusionner les 4 mots de `live` en une seule boucle ramènerait 33,2 tours à
  24,8 — mais il faut alors avancer d'un mot à l'autre dans le corps, ce qui
  coûte les 4 instructions que l'on vient d'économiser. C'est un lavage exact.

**Le modèle « seul le compte d'instructions compte » (§4.8) tient toujours, et
il se retourne contre les optimisations mémoire.** Deux tentatives, mesurées,
toutes deux perdantes :

* **Balayage vectorisé.** Une tranche de bitmap tient en 16 octets ; en
  supprimant le bourrage anti-bancs (WP = W = 4) les W mots deviennent
  contigus et alignés, donc un seul `LDS.128` remplace 4 lectures 32 bits —
  64 lectures dynamiques tombent à 16 par thread et par vhi. **Mesure : 5 %
  plus lent** (trois tours). Le compte statique ne bouge pas (la boucle en `w`
  n'est pas déroulée), les registres passent de 64 à 80, et il faut garder les
  4 mots vivants pendant tout le drain. Ce qu'on gagne en lectures, on le perd
  en pression de registres.
* **Suppression des lectures partagées du drain.** Remplacer `sT[el2]` et
  `sPw[t2]` par de l'arithmétique sur `(t2, el2)` retire 13 LDS par survivant
  et rend le noyau **4,5 % plus lent** : la mémoire partagée n'est pas le
  goulot (LSU à ~15 %), et l'arithmétique de remplacement coûte plus cher que
  les lectures qu'elle évite.

Autrement dit : à ce stade, tout ce qui échange des instructions ALU contre du
trafic mémoire, ou l'inverse, est neutre ou perdant. Il ne reste que le compte,
et les deux gros blocs sont à leur plancher — l'arbre depuis la v6.1 (§4.9),
les popcounts depuis la v6 (§4.4).

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

**Tensor cores — fermés par la mesure.** Les 15 écarts impairs sont bilinéaires
en (o,e), donc formellement un GEMM : c'était la piste la plus prometteuse
restante. Elle ne tient pas, et l'erreur de l'estimation initiale est
identifiable.

Deux débits mesurés sur la 4070 de ce dépôt (`tensorcheck.cu`) :

| | débit |
|---|---|
| `mma.sync.m16n8k32.s8` (tensor cores INT8) | **116,0·10¹² MAC/s** |
| `xor` + masque + `popc` — le motif exact du drain | **46,8·10¹² MAC binaires/s** |

Le premier chiffre reproduit la spec (1,16·10¹⁴), donc le banc est bon. Le
rapport n'est que de **2,48×**, et c'est là que l'estimation initiale se
trompait : elle comparait implicitement le tensor core à de l'arithmétique
scalaire. Or **`popc` sur un XOR 32 bits *est* un produit scalaire binaire de
longueur 32** — une instruction SIMD-dans-un-registre. Le tensor core n'a pas
en face de lui un pipe scalaire, il a un autre moteur de produits scalaires.

Et il perd le reste sur le volume. Un GEMM calcule **tous** les couples
(thread, e_lo) d'un (bloc, vhi), soit 256 × 128 = 32 768 ; le drain, lui, ne
touche que les **12,9 %** qui survivent, et le §4.5 lui retire encore 8 des
30 corrélations.

| | MAC par (bloc, vhi) |
|---|---|
| GEMM dense : 32 768 couples × 675 positions | 22,1·10⁶ |
| drain actuel : 4 227 survivants × 519 positions | 2,19·10⁶ |

**10,1× plus de MAC à 2,48× la vitesse : le GEMM serait 4,1× plus lent.** Et ce
n'est même pas le pire : les 30 matrices de sortie font ~1 Mo par bloc, donc
elles ne tiennent pas en mémoire partagée. Il faudrait fondre le drain dans
l'épilogue du MMA, avec 30 fragments d'accumulateurs (120 registres) vivants
simultanément.

Reste la question naturelle : ne peut-on pas ne calculer que les survivants ?
Non. Chaque survivant a **son** o et **son** e ; un produit matriciel 16×8
calculerait les 128 couples croisés pour n'en garder que 16 — le gâchis passe
de 7,8× à 128×. La sparsité est un sous-ensemble de 12,9 % d'un produit
cartésien, exactement ce qu'un MMA dense ne sait pas exploiter.

### 5.3 Ouvertes, non explorées à fond

* **Battre le 4ⁿ.** Reste ouvert, et c'est désormais le **seul** point de cette
  section — les tensor cores et l'arithmétique du produit ont rejoint les pistes
  fermées. Les quatre voies connues pour casser l'exposant sont fermées aux
  §5.1 et §5.2 ; ce qu'il faudrait, c'est un mécanisme qui traite la contrainte de
  couverture (2^{2n} en inclusion-exclusion, 2ⁿ en largeur arborescente) et la
  contrainte de distinction des écarts (2ⁿ) **sans que les deux se multiplient**.
  La structure particulière du problème — la couleur d'une arête est déterminée
  par ses extrémités — n'a pas encore été exploitée pour cela.
* ~~**Arithmétique du produit.**~~ **Fermée par la v6.1** (§4.9). Les 21
  instructions de propagation de retenue ont disparu : les formes
  `mad{c}.{lo,hi}.cc` du PTX les absorbent dans le multiplieur, et l'étage final
  96×96→160 tombe à 17 instructions pour 17 produits partiels nécessaires —
  c'est-à-dire au plancher, une IMAD par produit et rien d'autre. Karatsuba y
  échangerait 2 multiplications contre une demi-douzaine d'additions : perdant.
  Reste l'arbre au-dessus (les 8 produits de feuilles et les 12 étages
  intermédiaires), qui est un arbre binaire équilibré et n'a pas de gras
  identifié.

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
| **ce dépôt (v7)** | 1 × RTX 4070 | **≈ 10,6 h** | **0,44** |

soit **~650× moins de GPU-jours**. Une part revient au matériel : un GPU Kepler
de 2013 délivre ~1,3·10¹² opérations entières/s contre 7,3·10¹² pour une 4070
(≈ 1,5·10¹³ en comptant le pipe FMA), soit un facteur **6 à 11**. Le reste —
**environ 50 à 90×** — vient de l'algorithme et de l'implémentation : symétrie
d'ordre 8 complète, saut des 87 % de produits nuls, décomposition de parité,
bitmaps de survie, et une arithmétique 160 bits tronquée plutôt que du CRT
modulaire. Chiffré directement à matériel identique (ci-dessous), cet écart
purement algorithmique vaut **~45×** ; les 50 à 90× le dépassent parce qu'ils
absorbent aussi le rendement de leur distribution sur ~32 cartes.

### Le point de départ vrai : l'état de l'art publié, sur la même carte

Le tableau du §4 part de ma **v3**, qui contient déjà deux choses absentes de
l'état de l'art publié. Pour savoir ce que ce dépôt fait gagner *à matériel
identique*, il faut chiffrer la version publiée sur la même 4070. Deux écarts,
et un seul est incertain.

* **La symétrie — exactement ×2, et ce n'est pas une estimation.** Assarpour
  *et al.* épinglent deux coordonnées (facteur 4) et n'exploitent pas la
  réflexion ; la v3 l'exploite et énumère 2^{2n−3} points au lieu de 2^{2n−2}.
  Le prédicat étant résolu au niveau du shard (§2.3), le coût par point ne
  bouge pas : le rapport est exactement 2.
* **L'arithmétique — ×4 sur le produit.** Ils accumulent modulo plusieurs
  premiers puis recombinent par CRT ; ici la somme tient dans 2¹⁶⁰ tronqué
  (§3.1). À n=31 elle vaut 2^{145,3}, donc il faut **5 premiers de 31 bits**,
  soit 5 × 31 multiplications modulaires par point contre un seul arbre
  160 bits de ~125 instructions (161 avant la v6.1, §4.9). Le facteur ~4 est
  celui du §6.bis ; le
  décompte d'instructions le confirme et donne sa fourchette — 2,5× pour une
  implémentation modulaire soignée (petits facteurs groupés par 4 avant
  réduction), 5× pour une implémentation directe.

Le produit pèse **70,6 % du temps de la v3** (§4.1), donc l'arithmétique
modulaire multiplie le coût par point par 0,294 + 0,706 × 4 = **3,12**, et la
symétrie manquante double le nombre de points :

| n=31, sur la même RTX 4070 | points | arithmétique | temps |
|---|---|---|---|
| Godfrey nu (2002), sans symétrie | 2⁶² | modulaire + CRT | ≈ 4 500 j |
| **état de l'art publié** (Assarpour *et al.*, 2015) | 2⁶⁰ | modulaire + CRT | **≈ 1 100 j — 3 ans** |
| même énumération, arithmétique de ce dépôt (v1/v2) | 2⁶⁰ | 160 bits tronqué | 360 j |
| v3 — point de départ du §4 | 2⁵⁹ | 160 bits tronqué | 180,1 j |
| v6 | 2⁵⁹ nominal | + bitmaps de survie | 32,2 j (mesuré) |
| v6.1 | 2⁵⁹ nominal | + arbre en chaînes de retenue | 29,2 j (mesuré) |
| **v7 — ce dépôt** | 2⁵⁹ nominal | + écarts impairs tabulés | **24,4 j (mesuré)** |

> **≈ 1 100 jours contre 24,4 mesurés : ~45× à matériel identique.**
> Fourchette 740 à 1 380 jours selon la qualité de l'implémentation modulaire,
> soit **30× à 57×**. Le 5,30× du §4 n'en est que la partie v3 → v6.
>
> Le **§4.0 décompose ce facteur étape par étape** — chaque optimisation avec
> son gain propre, le produit valant le total. Il l'exprime sur la base des
> débits nominaux (25,8 j pour la v7, donc ×42,6) plutôt que sur la mesure
> directe (24,4 j, donc ×45) ; l'écart de 6 % est dans les ±8 % d'erreur
> d'échantillonnage du §7.1.

Deux précautions. D'abord ce chiffrage est **favorable au point de comparaison** :
il lui prête mon code de Gray, mon empaquetage SWAR et mon arbre de produit, et
ne lui facture que l'arithmétique et la symétrie. Ensuite il se recoupe avec le
seul point de mesure réel : appliqué à n=28 — où 4 premiers suffisent, donc un
facteur 5,11 au lieu de 6,25 — il donne 14,4 j sur une 4070, soit **86 à
158 GPU-jours Kepler** avec le facteur matériel de 6 à 11× ; le calcul publié
en a coûté ~288. Le modèle est donc **conservateur d'un facteur 1,8 à 3,3**,
l'écart restant étant le rendement de leur distribution sur ~32 cartes.

**Sur le record lui-même : non, pas encore.** L(2,31) reste inconnu. Ce qui a
changé, c'est son prix — colonne « avant » = ma v3 ; l'état de l'art publié, lui,
demanderait ≈ 1 100 jours sur la même carte :

| | avant (v3) | maintenant (v7) |
|---|---|---|
| sur une 4070 | 180 jours | **25,8 jours** |
| sur une RTX 4090 | ~67 jours | **~9,5 jours** |
| en location grand public (~0,35 $/h) | ~560 $ | **~80 $** |
| en parallèle | ~26 GPU pendant une semaine | **~3,7 GPU pendant une semaine** |

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
| v6, mesurée | 34,0 j |
| v6.1, mesurée | 30,9 j |
| **v7, mesurée** | **25,8 j** |
| même code à 100 % d'émission (inatteignable) | 17,1 j |

À la v6.1 : 37,4 instructions par point canonique (0,129 × 251 de drain, plus
~5 de base), 216,3 Gsums/s canoniques → 8,09·10¹² instructions/s contre un
plafond d'émission de 1,46·10¹³, soit **55 % du plafond de la carte**.

**La v7 casse deux affirmations de ce paragraphe, et il faut le dire.**

1. *« Les 22 popcounts des écarts impairs sont exactement au minimum. »* Vrai
   pour le calcul tel qu'il était posé, faux pour le problème : la v7 n'en
   exécute **aucun** dans le drain (§4.10). Le minimum d'une formulation n'est
   pas le minimum du calcul — c'est la leçon la plus utile de cette passe.
2. *« Seul le nombre d'instructions compte encore. »* Deux mesures le
   contredisent maintenant, dans les deux sens : une réécriture qui retire
   16 instructions du noyau et coûte **3,5 %** (§4.6, constantes
   pré-additionnées), et le chemin rapide du produit qui en **ajoute** au
   compte statique tout en gagnant 7,4 % (§4.10). Le modèle du §4.8 reste une
   bonne première approximation ; il ne prédit plus le signe du gain à lui
   seul.

Ce qui survit, en revanche : ni l'occupancy, ni les pipes, ni la mémoire ne
sont le facteur limitant — et l'occupancy a été retestée sur la v7, avec cette
fois une réponse **nette** dans l'autre sens (2 blocs par SM : 27 % plus lent,
§4.10). Le noyau reste limité par ce que son propre mélange d'instructions lui
impose à l'émission.

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
* **Chaînes de retenue `mad{c}.{lo,hi}.cc` du PTX** (§4.9) : l'idiome standard
  des bibliothèques de grands entiers sur GPU (CGBN, et l'exemple `mp` du SDK).
  Ce qui est à moi ici n'est pas l'idiome, c'est d'avoir mesuré ce qu'il vaut
  sur *ce* noyau et d'avoir vérifié que ptxas ne le trouve pas tout seul.
* **Extension de signe d'octet par `prmt.b32`** avec le bit 3 du sélecteur
  (§4.9) : documenté dans l'ISA PTX, mais nvcc ne l'émet que pour l'octet 0.
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
6. **Décomposition des écarts impairs sur l'axe `threadIdx`** (§4.10). Les 15
   écarts impairs sont bilinéaires en (o,e) : ils ne se séparent pas comme les
   pairs, et c'est pour cela que le drain les recalculait par 22 popcounts et
   par survivant. Ils se séparent quand même, mais sur un axe qui n'a rien à
   voir avec la parité : parce que `o` est construit par `__brev`, `threadIdx`
   ne pilote que **huit** bits de `o`, et tout le reste est constant sur le
   bloc. En classant les couples (a, c) d'un écart selon que *a* tombe dans
   cette fenêtre et *c* dans `e_lo`, il ne reste que trois termes
   indissociables à n=31 : les écarts impairs deviennent une somme de deux
   tables, exactement comme les pairs. La tabulation est une technique banale ;
   l'axe de décomposition, je ne l'ai vu nulle part. **+19,7 % mesuré.**
7. **Le tensor core ne bat pas le popcount sur une corrélation ±1** (§5.2).
   Le réflexe « c'est bilinéaire, donc c'est un GEMM, donc ça va vite » est
   faux ici, et pour une raison qui se mesure : `popc` sur un XOR 32 bits *est*
   un produit scalaire binaire de longueur 32, si bien que l'INT8 ne mène que
   2,48× sur cette carte — écart que la sparsité du problème (12,9 % de
   survivants, qu'un MMA dense ne sait pas exploiter) retourne en 4,1× de
   retard. Le résultat est négatif mais il est net, et je ne l'ai vu chiffré
   nulle part.

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

> **RTX 4070 : n=31 en 24,4 jours (586 h GPU), IC95 % [528, 645].**
> Le chiffre de 25,8 j annoncé plus haut vient du débit nominal en Gsums/s ;
> la mesure directe est un peu meilleure. C'est celle qui fait foi.
>
> (v6 : 32,2 j / 772 h ; v6.1 : 29,2 j / 701 h. La v7 est mesurée **appariée**
> contre la v6.1, à +19,7 % sur quatre tours et deux échantillons — +21,00 /
> +19,64 / +19,55 / +19,53 %, §4.10 ; c'est ce rapport qui est fiable, et il
> est appliqué ici au 701 h de référence. Les valeurs absolues, elles, portent
> ±8 % : le même binaire v6 mesure 720 h ou 772 h selon l'échantillon et l'état
> d'horloge de la carte. La v7, elle, est reproductible à 0,02 % d'un tour à
> l'autre — la disparition des popcounts a rendu le coût bien moins dépendant
> du motif de bits.)

La graine du tirage est **fixe**. Deux GPU différents mesurent donc le même
échantillon de `vhi` : la comparaison est appariée, et le rapport entre deux
cartes est bien plus précis que chacune des deux estimations absolues.

| n | v3 | v4 | v5 | v6 | v6.1 | **v7** |
|---|---|---|---|---|---|---|
| 24 | 15,9 min | 9,0 min | 6,6 min | 3,0 min | 2,8 min | **2,8 min** (+1,5 %) |
| 27 | 16,9 h | 9,5 h | 7,0 h | 3,18 h | ≈ 2,95 h | **2,48 h** (total calculé) |
| 28 | 2,82 j | 1,59 j | 1,17 j | 12,7 h | ≈ 11,7 h | **≈ 10,6 h** (+10,2 %) |
| **31** | 180,1 j | 101,7 j | 75,0 j | 32,2 j | 29,2 j | **24,4 j** (+19,7 %) |
| 32 | 1,97 an | 1,11 an | 0,82 an | ≈ 129 j | ≈ 117 j | **≈ 96 j** |

Colonne v7 : le gain est mesuré **à chaque n** — n=24 bout en bout (186,8 s
contre 184,1 s), n=28 et 31 par `--bench` apparié — et non interpolé, parce
qu'il varie beaucoup avec *n* (§4.10). **n=27 n'est pas une projection du tout :
le total complet a tourné, en 8 915 s (2 h 28), à 505,1 Gsums/s** — 5 % plus
vite que les 2,61 h projetés, et cela *malgré* une machine qui faisait tourner
`slice_ref` sur 18 cœurs pendant une partie du run. La ligne n=32 reste, elle,
une extrapolation du rapport de n=31.

Tout l'état de l'art publié (L(27) + L(28)) se refait en **~13,1 heures** sur
cette carte — dont la moitié est désormais faite : L(27) a été recalculé ici en
2 h 28 et concorde à l'unité près (§3.2d).

### 7.2  Quelle architecture — le modèle, puis la mesure qui le contredit

Le drain du noyau exécute **161 instructions ALU entières pour 95 IMAD** et
~32 autres, soit 288 instructions-warp. D'où ce raisonnement :

*(Chiffres de la v6 ; la v6.1 ramène le drain à 251 instructions et déplace le
mélange vers l'IMAD — 95 IADD3 et 455 IMAD sur le noyau entier, contre 149 et
429. La ligne « goulot ALU » du tableau ci-dessous en devient moins serrée, ce
qui va dans le même sens que la mesure qui suit : l'avantage prédit pour
Blackwell rétrécit encore.)*

| architecture | voies INT32 / SM | cycles ALU | cycles d'émission | goulot | perf / SM·cycle |
|---|---|---|---|---|---|
| Ampere GA10x | 64 | 161/2 = 80,5 | 288/4 = 72 | **ALU** | 1,00 |
| Ada AD10x | 64 | 80,5 | 72 | **ALU** | 1,00 |
| Hopper H100 | 64 | 80,5 | 72 | **ALU** | 1,00 |
| Blackwell GB20x | **128 unifiées** | (161+95)/4 = 64 | 72 | **émission** | **1,12** |

Blackwell fusionne les cœurs INT32 et FP32, donc le noyau devait cesser d'être
limité par l'ALU et buter sur l'émission — d'où 1,12, et un rapport 5090/4070
prédit de 3,70 × 0,97 × 1,12 = **4,02**.

> **Mesuré sur une RTX 5090 louée le 2026-09-04 : 3,05.**
> Le modèle était **32 % trop optimiste**.

Le protocole : même travail exactement des deux côtés (mêmes `vhi`, un seul
processus par carte pour que l'initialisation du contexte CUDA — 0,12 s sur la
4070, 0,23 s sur l'instance louée — reste négligeable). Trois régions :

| `vhi` | 4070 | 5090 | rapport |
|---|---|---|---|
| 0, 24 valeurs | 50,24 s | 16,92 s | 2,97 |
| 2 000 000, 24 valeurs | 11,18 s | 3,91 s | 2,86 |
| 5 000 000, 40 valeurs | 9,20 s | 3,26 s | 2,82 |

puis 3,05 après le réglage d'occupancy du §7.7.

**Ce n'est pas du throttling** : relevé sous charge, la 5090 tient 2715 MHz à
575 W, 100 % d'utilisation, 78 °C — au-dessus de son boost nominal. Ramené au
SM et au cycle, elle est donc à **0,82× la 4070**, et non 1,12. Le doublement du
débit entier annoncé par Blackwell ne se matérialise pas pour ce noyau. Le
tableau ci-dessus reste la meilleure explication que j'aie du comportement
d'Ada ; pour Blackwell il est simplement faux, et je ne sais pas dire pourquoi.

Ce qui survit à la mesure, en revanche, c'est la conclusion sur les cartes de
datacenter : H100 et A100 ont les mêmes 64 voies INT32 par SM qu'une carte grand
public pour 6 à 13 fois le prix horaire, et leurs cœurs tensoriels comme leur
HBM ne servent à rien ici. « Ne servent à rien » est maintenant chiffré et non
supposé : un GEMM INT8 sur les écarts impairs serait **4,1× plus lent** que les
popcounts qu'il remplacerait (§5.2). Louer du tensor core pour ce noyau, c'est
payer un silicium qu'il n'utilisera pas.

### 7.3  Coût réel sur vast.ai (relevé septembre 2026)

`h GPU` = 545 / rapport, où 545 h est la mesure directe de la v7 sur la 4070
(`--bench 96`). Le coût total ne dépend **que** des heures-GPU : la
parallélisation n'achète que du temps de calendrier, jamais des euros. Les
lignes 4090 et 5090 reposent désormais **toutes deux sur une mesure** ; les
autres rapports restent modélisés.

| GPU | rapport | h GPU | $/h spot | **coût spot** | $/h à la demande | coût |
|---|---|---|---|---|---|---|
| RTX 4070 (référence) | 1,00 | 545 | — | — | — | — |
| RTX 3090 | 1,22 *(modèle)* | 447 | 0,12 | 54 $ | 0,20 | 89 $ |
| **RTX 4090** | **2,66 *(mesuré)*** | **205** | 0,11 | **23 $** | 0,25 | 51 $ |
| RTX 5090 | 3,05 *(mesuré sur la v6.1)* | 179 | 0,20 | 36 $ | 0,336 | 60 $ |
| H100 SXM | 2,03 *(modèle)* | 268 | — | — | 1,65 | 443 $ |

> **RTX 4090 : rapport 2,66, mesuré le 2026-09-05.** Le §7.2 le modélisait à
> 2,83 ; l'écart est de 6 %, très loin des 32 % du cas Blackwell. Protocole :
> même binaire (le `.fat` sm_89+sm_120), même échantillon de `vhi`
> (`--bench 96`, graine fixe des deux côtés), carte vérifiée libre avant la
> mesure (1 Mo occupé, 0 % d'utilisation — le piège des cartes sur-souscrites
> du §7.7). 4070 : 545,1 h / 0,2339 s/vhi. 4090 : 204,8 h / 0,0879 s/vhi.
>
> **Et la réponse à la question ouverte du §7.3 est oui** : à prix spot la 4090
> est le choix le moins cher (23 $ contre 36 $), parce que son rabais spot est
> plus profond que son déficit de débit. La 5090 reste le meilleur choix si
> c'est le temps de calendrier qui compte.
>
> Réserve honnête : le rapport 5090 de 3,05 a été mesuré **sur la v6.1**. La v7
> a changé le mélange d'instructions (plus de `prmt` et de `LDS.64`, plus aucun
> `popc` dans le drain) ; le 2,66 de la 4090, lui, est mesuré sur la v7. Les
> deux lignes ne sont donc pas strictement comparables.

La 5090 reste le meilleur choix mesuré, mais l'écart avec la 4090 se resserre
nettement une fois le modèle corrigé : 3,05 contre 2,83, pour un prix spot
presque double. **Si le rapport 2,83 de la 4090 se confirmait par la mesure,
elle serait le choix le moins cher** — cela vaut le benchmark à 0,14 $.

**Sur une seule 4090 : 204,9 h GPU, soit 8,5 jours.** Le coût ne dépendant que
des heures-GPU, il reste **~23 $ en spot** quel que soit le nombre de cartes ;
n'achète du parallélisme que le temps de calendrier.

| 4090 en parallèle | 1 | 4 | 8 | **17** | 25 | 50 |
|---|---|---|---|---|---|---|
| calendrier (avec la 4070 locale) | 6,2 j | 46,8 h | 24,5 h | **11,8 h** | 8,1 h | 4,1 h |

**Pour un résultat en 12 heures : 17 cartes suffisent au nominal, 19 sont
prudentes.** Le total de 545 h porte ±8 % (§7.1), et c'est cette incertitude qui
décide, pas l'arrondi :

| hypothèse sur le total | cartes pour tenir 12 h |
|---|---|
| optimiste, 501 h (−8 %) | 16 |
| **mesuré, 545 h** | **17** |
| pessimiste, 589 h (+8 %) | **19** |

| cartes | calendrier si 545 h | si 589 h |
|---|---|---|
| 17 | 11,8 h | 12,7 h — *dépasse* |
| 18 | 11,2 h | 12,0 h |
| **19** | **10,6 h** | **11,4 h** |
| 20 | 10,1 h | 10,9 h |

Trois choses que ces tableaux ne disent pas et qui coûtent du calendrier :
une instance spot préemptée reperd sa tâche en cours (~3 min avec T = 4096) ;
chaque location met ~4 min à démarrer avec l'image `base` (§7.4) ; et une carte
sur-souscrite mesure 0,85× une 4070 au lieu de 2,66× (§7.7) — `provision()` la
refuse, mais elle a été payée. **19 cartes** absorbent les trois.

**Et si l'on veut le résultat défendable plutôt que rapide** (§3.3bis), il faut
dupliquer le run : **34 cartes pour 11,9 h**, ~45 $ en spot. C'est le seul
scénario que je recommanderais pour un chiffre destiné à être publié.

*Le rapport 5090/4070 de 3,05 reste celui de la v6.1 ; il mériterait d'être
remesuré sur la v7, c'est le même benchmark à 0,03 $.*

### 7.4  Plan recommandé

1. **Louer une seule carte spot dix minutes (≈ 0,03 $)** et lancer
   `./langford6 -n 31 --bench 96`. C'est ainsi qu'ont été obtenus les deux
   rapports mesurés du §7.3 (5090 : 3,05 sur la v6.1 ; 4090 : 2,66 sur la v7),
   et c'est ce qui reste à faire pour la 5090 sur la v7. Tout le reste du
   budget en découle.

   Deux pièges rencontrés, tous deux coûteux en temps :
   * **louer ne démarre pas.** Une instance vast naît avec
     `intended_status=stopped` : elle télécharge son image puis reste là, et son
     port SSH refuse la connexion indéfiniment. `orchestrator.py up` demande
     maintenant le démarrage explicitement ; sans cela, l'attente est infinie.
   * **l'image `devel` coûte plus cher que le calcul.** Ses 6 Go mettent plus de
     25 minutes à se télécharger sur certains hôtes. Le binaire préfabriqué
     `langford6.fat` étant lié en statique (`-cudart static`, sm_89 + sm_120),
     l'image `base` de 200 Mo suffit : instance prête en ~3 minutes.
     `up --image devel` remet l'ancienne si l'on veut compiler sur place. Elle
     doit être en **ubuntu24.04** : la 22.04 (glibc 2.35) refuse un `.fat`
     compilé sur une machine récente.
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

### 7.6  Orchestration : louer, distribuer, reprendre

`orchestrator.py` pilote un calcul complet depuis la machine locale. Trois
contraintes ont dicté la conception.

**Le maître initie toutes les connexions.** Une machine domestique est derrière
un NAT : les instances louées ne peuvent pas l'appeler. Le maître pousse donc le
travail par SSH, et la clé publique est injectée à la création via le script
`onstart` de vast.ai — ce qui évite d'avoir à gérer des clés au niveau du compte.

**L'état survit à tout.** SQLite en `journal_mode=WAL` + `synchronous=FULL`, et
chaque résultat est validé dès son arrivée — pas à la fin d'un lot. Un `kill -9`
du maître ne coûte que les tâches en vol. Au redémarrage, comme le maître est le
*seul* distributeur (verrou `flock`), on sait que rien ne tourne : les baux
hérités sont libérés immédiatement au lieu d'attendre leur expiration.

**Le travail est distribué par baux, pas par tranches fixes.** Un worker
demande un lot, on le lui réserve pour une durée bornée. S'il ne rend rien à
temps — préemption spot, machine perdue, SSH coupé — le bail expire et le lot
retourne au pot. Aucune tâche ne peut être définitivement perdue, et un worker
lent ne bloque personne. Le lot est dimensionné d'après le débit *mesuré* de
chaque worker pour durer ~6 min, et `run_batch.sh` émet chaque résultat au fil
de l'eau : une coupure ne perd jamais plus que la tâche en cours.

La 4070 locale est un worker comme les autres, simplement sans SSH.

```sh
./orchestrator.py init -n 31 -T 8192     # 8192 tâches à charge égale
./orchestrator.py plan --hours 10        # combien de 5090 pour tenir en 10 h
./orchestrator.py offers                 # le marché en direct
./orchestrator.py up --count 19          # louer (ou --bid 0.25 pour du spot)
./orchestrator.py bench                  # mesurer le vrai rapport 5090/4070
./orchestrator.py run                    # boucle principale, reprenable
./orchestrator.py status                 # avancement, par worker
./orchestrator.py merge                  # somme + auto-test + réponse
./orchestrator.py down                   # tout détruire
```

**Ce qui est vérifié et ce qui ne l'est pas.** Le cœur — découpage, baux,
reprise après `kill -9`, dispatch multi-worker, agrégation — est validé de bout
en bout sur n=20 et n=24, résultats exacts. En revanche `up`, `down` et le
provisionnement SSH n'ont **jamais pu s'exécuter** : la clé API disponible
renvoie 401 sur tout endpoint authentifié (vast.ai exige une clé créée depuis
une session 2FA ; seule la recherche d'offres est publique). Ce chemin est donc
écrit mais non testé contre l'API réelle.

**Rendre la clé utilisable** : se reconnecter sur cloud.vast.ai *via la 2FA*,
page Keys → +New (full-access par défaut), puis remplacer la valeur dans `.env`.
Activer la 2FA ne débloque pas rétroactivement une clé créée avant.

`.env` est en `chmod 600` et exclu par `.gitignore`, comme `state.db`, la paire
de clés SSH générée et les sommes partielles.

### 7.7  Ce que la première location a appris

Une seule RTX 5090 louée 40 minutes, pour **0,14 $ au total**, a corrigé trois
choses qu'aucun raisonnement n'aurait trouvées.

**Le rapport annoncé était faux de 32 %** (§7.2). C'est la raison d'être du
benchmark à 0,14 $ avant d'engager 50 $ : le dimensionnement passe de 19 à
25 instances pour tenir en 10 h.

**Le noyau débordait des registres sur sm_120.** À 5 blocs/SM — le réglage
optimal sur Ada — ptxas signale 20 octets de *spill stores* sur Blackwell, et
40 à 6 blocs. Balayage sur les deux cartes :

| blocs/SM | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| 5090 | 3681 ms | **3676 ms** | 3813 | 3931 *(spill 20 o)* | 3967 *(40 o)* |
| 4070 | 11143 ms | 11199 ms | 11280 | 11145 | 11290 *(spill 28 o)* |

**3 blocs/SM** supprime le débordement et gagne 6,5 % sur Blackwell, en restant
dans le bruit sur Ada — c'est désormais le défaut (`LF6_MINBLK`, redéfinissable
par carte). Gain net : 6,5 % de la facture.

**`--bench` mentait, et de deux façons.** Il annonçait un rapport de 5,8×
là où le travail identique donnait 3,05×. Deux causes :

* le tirage LCG dégénérait avec le nombre d'échantillons — juste à 64 tirages
  (778 h contre 772 h réelles), deux fois trop pessimiste à 128 (1589 h). Les
  bits de poids faible d'un LCG à module 2^32 ont des périodes très courtes ;
  remplacé par splitmix64 ;
* il chronométrait un lancement isolé à `count=1`, qui ne charge pas deux
  cartes de la même façon. Il mesure maintenant le chemin de **production** :
  même `count` qu'un `--chunk`, recopie et réduction hôte comprises.

Après correction, l'outil recoupe la mesure directe : 48 tirages donnent 728 h
contre 772 h de vérité terrain, et le rapport apparié 4070/5090 ressort à
**3,13**, contre 3,05 par le travail identique — 3 % d'écart, là où l'ancienne
version annonçait 5,8.
La leçon générale rejoint le §4.7 : sur ce problème, **tout estimateur qui
n'échantillonne pas uniformément les `vhi` ment**, et le rapport de 409 entre
le `vhi` le plus cher et le moins cher fait qu'il ment beaucoup.

**Une offre bon marché peut être une carte sur-souscrite, et rien ne le dit.**
Une « RTX 4090 à 0,136 $/h » mesurait **0,85× une 4070** — plus lente qu'une
carte trois fois plus petite. Le matériel semblait pourtant sain : 2670 MHz,
441 W sur 450, PCIe gen4 x16, 100 % d'utilisation, et CUDA voyait bien les
128 SM. L'explication n'apparaît qu'en interrogeant la carte **à l'arrêt** :

```
$ nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv
20531 MiB, 100 %, 427.41 W        <- alors que notre calcul ne tourne pas
$ nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
GPU-3a0ffed4…, 2704705, 4126 MiB
GPU-3a0ffed4…, 3010397, 5470 MiB   <- cinq autres processus,
GPU-3a0ffed4…, 3673338, 6226 MiB      sur le meme UUID que le notre
GPU-3a0ffed4…,  898865, 1394 MiB
GPU-3a0ffed4…,  994518, 3266 MiB
```

Le prix était le seul indice, et il ne suffit pas. `provision()` interroge donc
maintenant la carte avant d'accepter la machine et l'écarte au-delà de 2 Go
occupés ou 25 % d'utilisation : payer un sixième de GPU pendant dix heures coûte
bien plus cher que de jeter l'instance tout de suite. **Toute mesure de
comparaison entre cartes est sans valeur si ce contrôle n'a pas été fait** — et
je ne l'avais pas fait sur la 5090, dont le 3,05 mesuré pourrait donc être un
plancher.

**Le provisionnement coûte plus cher que prévu.** L'image `nvidia/cuda:…-devel`
pèse ~6 Go et son téléchargement est facturé comme du calcul. Sur un hôte à
213 Mbps, une instance est restée dix minutes sans jamais ouvrir son SSH et a dû
être détruite ; sur 25 instances, ce temps mort n'est plus anecdotique. D'où
deux corrections : `up --min-net` écarte les hôtes mal connectés (défaut
400 Mbps), et le provisionnement envoie désormais un **binaire prefabriqué de
2 Mo** compilé pour sm_89 *et* sm_120 — cudart lié en statique, aucune
dépendance CUDA dynamique — au lieu de lancer `nvcc` sur chaque machine. Il est
vérifié sur la carte distante (il doit retrouver `L(2,12) = 108144`) avant
d'être accepté, sinon on retombe sur la compilation.

**Ce qui a marché du premier coup**, en revanche : injection de la clé SSH par
le script `onstart`, transfert des sources, compilation nvcc en CUDA 12.8 pour
sm_120, et destruction de l'instance. Le chemin `up` → provisionnement → `down`
n'avait jamais pu s'exécuter avant faute d'accès API.

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
arrêté. Pour un calcul piloté de bout en bout, voir `orchestrator.py` (§7.6).
Les briques de plus bas niveau, si besoin :

```sh
./run_shard.sh 17 4096 31        # une tâche isolée -> ligne PART=...
./langford6 -n 31 --diag         # les orbites fixes (une seule fois)
./langford6 -n 31 --merge <PART...>   # somme + auto-test de divisibilité
./langford6 -n 31 --merge-file f.txt  # idem, depuis un fichier (>1000 tranches)
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

* `langford6.cu` — **version de référence (v7)** : coordonnées de parité,
  bitmaps de survie, réflexion reparamétrée, produit en complément à deux,
  arbre 160 bits en chaînes de retenue PTX et extraction SWAR par `prmt`
  (§4.9), écarts impairs tabulés en somme de deux mots SWAR (§4.10).
  `-DK_=<k>` recompile avec un autre découpage `e_lo` ; K=7 est l'optimum
  mesuré sur la v6.1, **non rebalayé depuis la v7** — voir §4.10
* `check_decomp.c` (§9, vérification) est le contrôle qui valide la
  décomposition sur laquelle repose ce noyau
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
* `run_batch.sh` — exécute un lot de tâches, émet chaque résultat au fil de l'eau
* `orchestrator.py` — location vast.ai, distribution par baux, état SQLite
  reprenable (§7.6)
* `collect.sh` — vérifie complétude et absence de doublons, puis conclut

**Vérification et théorie**

* `tensorcheck.cu` — débit mesuré des tensor cores INT8 contre le motif
  `xor`+masque+`popc` du drain : c'est ce banc qui ferme la piste GEMM (§5.2)
* `verify_all.sh` — **rejoue toute la chaîne de validation** en une commande
  (§3.5) ; sort non nul si quoi que ce soit diverge
* `ladder.sh` — l'échelle **n = 1 à 24**, n par n : valeur attendue, mode de
  couverture, et vérification (`GPU=0` pour la partie CPU seule)
* `verify.c` — identité de Godfrey contre une **force brute** indépendante,
  n = 1..16, et la condition de validité de la réduction ×4 (§3.2a)
* `cover_check.c` — le **pont preuve ↔ code** : la couverture exacte vérifiée
  exhaustivement sur l'indexation réelle du noyau, jusqu'à 1,07·10⁹ index à
  n=31 (§3.2b)
* `slice_ref.c` — recalcule une tranche `PART=` par l'**algorithme classique**
  (Godfrey nu, sans parité ni SWAR ni PTX ; OpenMP) ; `check_slices.sh` compare
  au bit près (§3.2c)
* `refvals.txt` / `check_refs.sh` — les valeurs de référence du §3.2c, et leur
  vérification automatique par le GPU ou par l'algorithme classique
* `audit.sh` — dossier d'audit d'une campagne : complétude, empreintes des
  binaires, cartes, et **recalcul redondant d'un échantillon** (§3.6)
* `proof/Langford.lean` — **preuve Lean 4** que l'énumération du noyau couvre
  chaque point exactement une fois : involutivité de `f` et de σ, unicité du
  translaté épinglé de Klein (facteur 4 exact), trichotomie de la réflexion
  (poids 2 hors diagonale, 1 dessus). Sans mathlib, sans `sorry` ; `#print
  axioms` ne rend que `propext` et `Quot.sound`. Portée et limites dans
  `proof/README.md`
* `oe_check.c` — décomposition de parité, vérifiée jusqu'à n=31
* `check_decomp.c` — décomposition v7 des écarts **impairs** en
  `Base(o,e_hi) + delta(bloc,e_lo) + termes croisés`, vérifiée terme à terme et
  qui énumère les termes croisés restants (§4.10)
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
* `run_shard.sh` / `worker.sh` émettent la **provenance** de chaque tâche
  (empreinte du binaire, carte, pilote, durée, horodatage) — c'est ce que
  relit `audit.sh`
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
