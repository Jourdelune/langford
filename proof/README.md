# Preuve Lean : l'énumération du noyau est exacte

`Langford.lean` prouve, en **Lean 4.15 (noyau seul, sans mathlib)**, que
`oe_kernel` + `diag_kernel` visitent chaque point de l'espace de Godfrey
**exactement une fois**, avec les poids annoncés.

```sh
lean Langford.lean          # aucune sortie = tout passe
```

## Pourquoi cette propriété-là

Les auto-tests du dépôt (§3 du README principal) attrapent une tranche
corrompue, dupliquée ou manquante. Ils n'attrapent **pas** une erreur
systématique de couverture : si le prédicat canonique oubliait une orbite, ou
en comptait une deux fois, toutes les tranches seraient « valides » et
L(2,31) serait faux en silence. Les huit valeurs connues couvrent n ≤ 24 ;
rien ne garantissait le raisonnement lui-même.

## Ce qui est prouvé

| théorème | énoncé |
|---|---|
| `f_involutive` | `f = canon ∘ rev` est une involution sur les rangées épinglées |
| `sigma_involutive` | la réflexion σ(o,e) = (f e, f o) est une involution sur les points épinglés |
| `sigma_pinned` | σ préserve l'épinglage |
| `klein_exact` | tout point a **exactement un** translaté épinglé par le groupe de Klein |
| `klein_orbit_card_four` | les 4 translatés de Klein sont **distincts** — le facteur est 4, pas moins |
| `diag_fixed` | un point diagonal (`e = f o`) est un point **fixe** de σ : d'où son poids 1 |
| `reflexion_exact` | pour tout point épinglé, **exactement un** des trois cas : énuméré / diagonal / miroir énuméré |
| `sigma_rescues` | un point épinglé non énuméré a son miroir énuméré : rien n'est perdu |
| `couverture_exacte` | **tout** point possède un représentant énuméré |

Aucun `sorry`. `#print axioms` sur les huit théorèmes ne rend que `propext` et
`Quot.sound`, les axiomes standard du noyau de Lean — donc aucune brèche.

Le théorème sur la réflexion est prouvé pour **un ordre total strict
quelconque** sur les rangées, pas seulement l'ordre entier sur le motif de
bits qu'utilise le noyau. Il est donc plus fort que nécessaire, et robuste à
un changement de codage.

## Ce qui n'est PAS prouvé ici, et qu'il faut dire

1. **L'identité de Godfrey elle-même** — que la somme pondérée compte bien les
   suites de Langford. C'est le théorème de 2002 ; il est ici corroboré par
   les huit valeurs connues reproduites à l'unité près, pas formalisé.
2. **La décomposition de parité** (écarts pairs séparables, impairs
   bilinéaires) : vérifiée exhaustivement par `oe_check.c` jusqu'à n=31, et la
   décomposition v7 des écarts impairs par `check_decomp.c`.
3. **L'arithmétique 160 bits** du noyau : corroborée par les sommes partielles
   identiques au bit près entre versions.
4. **Le pont modèle → code.** La preuve porte sur des rangées `Fin (n+1) →
   Bool` ; le noyau les représente par des masques 32 bits, et son prédicat
   `v < u` correspond à `e < f o` parce que `f` est involutive et que
   `o = f(F)` avec `F = u<<1`. Cette correspondance est un argument de lecture
   du code, pas un théorème mécanisé. C'est le maillon qui reste humain.

Autrement dit : la preuve ferme le trou du *raisonnement de symétrie*. Les
trois autres maillons restent couverts par la mesure, comme avant.
