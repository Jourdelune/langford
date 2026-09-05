/-
  Langford L(2,n) -- la couverture de l'enumeration est EXACTE.

  Ce fichier prouve, en Lean 4 (noyau seul, sans mathlib), que le noyau
  `oe_kernel` + `diag_kernel` visite chaque point de l'espace de Godfrey
  exactement une fois, avec les poids annonces.  C'est la propriete dont
  depend la justesse du resultat : si un seul point etait oublie ou compte
  deux fois, L(2,n) serait faux sans qu'aucun auto-test ne s'en apercoive.

  Ce qui est formalise ici est la COMBINATOIRE DE L'ENUMERATION :

    * l'espace : un point est un couple de rangees (o, e) de n+1 cases ;
    * le groupe de Klein : negation independante de chaque rangee ;
    * la reflexion sigma(o,e) = (f e, f o), avec f = canon o rev ;
    * l'enumeration du noyau : on epingle o_0 = e_0 = 0 (facteur 4), puis on
      ne garde que les points avec e < f o (poids 2, `oe_kernel`), la
      diagonale e = f o etant traitee a part avec le poids 1 (`diag_kernel`).

  Les quatre theoremes finaux (`klein_exact`, `klein_orbit_card_four`,
  `reflexion_exact`, `couverture_exacte`) disent ensemble : tout point a
  exactement un representant enumere, et les poids 4 puis 2/1 sont les bons.

  Ce qui n'est PAS formalise ici, et qu'il faut dire : l'identite de Godfrey
  elle-meme (que la somme ponderee compte bien les suites de Langford), et
  l'arithmetique 160 bits du noyau.  La premiere est un theoreme de 2002
  verifie ici par les huit valeurs connues ; la seconde par les sommes
  partielles identiques au bit pres.  Ce fichier ferme le troisieme trou :
  celui de l'enumeration.
-/

namespace Langford

/-- Une rangee : `n+1` cases booleennes.  Le `+1` garantit que l'indice 0,
    celui qu'on epingle, existe toujours. -/
abbrev Row (n : Nat) := Fin (n + 1) → Bool

variable {n : Nat}

/-- Negation d'une rangee : c'est l'action de X -> -X restreinte a une rangee. -/
def neg (r : Row n) : Row n := fun i => !(r i)

/-- Renversement d'une rangee : la reflexion p -> 2n+1-p en coordonnees de parite. -/
def rev (r : Row n) : Row n := fun i => r i.rev

/-- Une rangee est *epinglee* quand sa case 0 vaut `false`. -/
def pinned (r : Row n) : Prop := r 0 = false

instance (r : Row n) : Decidable (pinned r) := by unfold pinned; infer_instance

/-- Canonicalisation : on nie la rangee si sa case 0 vaut 1. -/
def canon (r : Row n) : Row n := if r 0 then neg r else r

/-- `f = canon o rev`, l'application dont le README dit qu'elle est involutive. -/
def f (r : Row n) : Row n := canon (rev r)

/- ------------------------------------------------------------------ -/
/- Lemmes de base                                                      -/
/- ------------------------------------------------------------------ -/

@[simp] theorem neg_neg (r : Row n) : neg (neg r) = r := by
  funext i; simp [neg]

@[simp] theorem rev_rev (r : Row n) : rev (rev r) = r := by
  funext i; simp [rev, Fin.rev_rev]

theorem rev_neg (r : Row n) : rev (neg r) = neg (rev r) := by
  funext i; simp [rev, neg]

@[simp] theorem neg_apply (r : Row n) (i : Fin (n+1)) : neg r i = !(r i) := rfl

/-- La negation ne fixe aucune rangee : elle change la case 0. -/
theorem neg_ne (r : Row n) : neg r ≠ r := by
  intro h
  have : (neg r) 0 = r 0 := by rw [h]
  simp [neg] at this

/-- `canon` produit toujours une rangee epinglee. -/
theorem canon_pinned (r : Row n) : pinned (canon r) := by
  unfold pinned canon
  cases h : r 0 <;> simp [h, neg]

/-- Sur une rangee deja epinglee, `canon` ne fait rien. -/
theorem canon_of_pinned {r : Row n} (h : pinned r) : canon r = r := by
  unfold canon; rw [h]; simp

/-- `f` produit toujours une rangee epinglee. -/
theorem f_pinned (r : Row n) : pinned (f r) := canon_pinned _

/-- **`f` est une involution sur les rangees epinglees.**
    C'est le fait dont depend toute la reparametrisation de la reflexion. -/
theorem f_involutive {r : Row n} (h : pinned r) : f (f r) = r := by
  unfold pinned at h
  cases hr : (rev r) 0 with
  | false =>
      have h1 : f r = rev r := by unfold f canon; rw [hr]; simp
      rw [h1]; unfold f canon; rw [rev_rev, h]; simp
  | true =>
      have h1 : f r = neg (rev r) := by unfold f canon; rw [hr]; simp
      rw [h1]; unfold f canon; rw [rev_neg, rev_rev]
      have h2 : (neg r) 0 = true := by simp [neg, h]
      rw [h2]; simp

/- ------------------------------------------------------------------ -/
/- Points, groupe de Klein, reflexion                                  -/
/- ------------------------------------------------------------------ -/

/-- Un point de l'espace de Godfrey en coordonnees de parite. -/
abbrev Point (n : Nat) := Row n × Row n

/-- Action du groupe de Klein : negation independante de chaque rangee. -/
def act (k : Bool × Bool) (p : Point n) : Point n :=
  ((if k.1 then neg p.1 else p.1), (if k.2 then neg p.2 else p.2))

/-- La reflexion : elle ECHANGE les deux rangees. -/
def sigma (p : Point n) : Point n := (f p.2, f p.1)

/-- Un point est epingle quand ses deux rangees le sont. -/
def pinnedP (p : Point n) : Prop := pinned p.1 ∧ pinned p.2

instance (p : Point n) : Decidable (pinnedP p) := by unfold pinnedP; infer_instance

/-- La reflexion preserve l'epinglage. -/
theorem sigma_pinned (p : Point n) : pinnedP (sigma p) :=
  ⟨f_pinned _, f_pinned _⟩

/-- **La reflexion est une involution sur les points epingles.** -/
theorem sigma_involutive {p : Point n} (h : pinnedP p) : sigma (sigma p) = p := by
  obtain ⟨h1, h2⟩ := h
  simp [sigma, f_involutive h1, f_involutive h2]

/- ------------------------------------------------------------------ -/
/- 1. Le facteur 4 : le groupe de Klein est exactement couvert         -/
/- ------------------------------------------------------------------ -/

/-- **Tout point a exactement un translate epingle par le groupe de Klein.**
    C'est ce qui justifie d'epingler o_0 = e_0 = 0 et de multiplier par 4 :
    aucune orbite n'est oubliee, aucune n'est comptee deux fois. -/
theorem klein_pin (p : Point n) : pinnedP (act (p.1 0, p.2 0) p) := by
  constructor
  · show (if p.1 0 then neg p.1 else p.1) 0 = false
    cases h : p.1 0 <;> simp [h, neg]
  · show (if p.2 0 then neg p.2 else p.2) 0 = false
    cases h : p.2 0 <;> simp [h, neg]

theorem klein_unique (p : Point n) (k : Bool × Bool) (hk : pinnedP (act k p)) :
    k = (p.1 0, p.2 0) := by
  obtain ⟨ha, hb⟩ := hk
  have e1 : k.1 = p.1 0 := by
    have : (if k.1 then neg p.1 else p.1) 0 = false := ha
    cases hk1 : k.1 <;> cases hp : p.1 0 <;>
      simp [hk1, hp, neg] at this ⊢
  have e2 : k.2 = p.2 0 := by
    have : (if k.2 then neg p.2 else p.2) 0 = false := hb
    cases hk2 : k.2 <;> cases hp : p.2 0 <;>
      simp [hk2, hp, neg] at this ⊢
  exact Prod.ext e1 e2

theorem klein_exact (p : Point n) :
    pinnedP (act (p.1 0, p.2 0) p) ∧
    ∀ k : Bool × Bool, pinnedP (act k p) → k = (p.1 0, p.2 0) :=
  ⟨klein_pin p, klein_unique p⟩

/-- **L'orbite de Klein d'un point a exactement 4 elements.**
    Le facteur est donc 4 et pas moins : les quatre translates sont distincts. -/
theorem klein_orbit_card_four (p : Point n) :
    ∀ k k' : Bool × Bool, act k p = act k' p → k = k' := by
  rintro ⟨a, b⟩ ⟨a', b'⟩ h
  have h1 : (if a then neg p.1 else p.1) = (if a' then neg p.1 else p.1) :=
    congrArg Prod.fst h
  have h2 : (if b then neg p.2 else p.2) = (if b' then neg p.2 else p.2) :=
    congrArg Prod.snd h
  have ea : a = a' := by
    cases a <;> cases a' <;> simp at h1 ⊢
    · exact absurd h1.symm (neg_ne p.1)
    · exact absurd h1 (neg_ne p.1)
  have eb : b = b' := by
    cases b <;> cases b' <;> simp at h2 ⊢
    · exact absurd h2.symm (neg_ne p.2)
    · exact absurd h2 (neg_ne p.2)
  simp [ea, eb]

/- ------------------------------------------------------------------ -/
/- 2. Le facteur 2 : la reflexion est exactement couverte              -/
/- ------------------------------------------------------------------ -/

section Reflexion

-- Un ordre strict quelconque sur les rangees.  Le noyau utilise l'ordre entier
-- sur le motif de bits, mais le theoreme ne depend PAS de ce choix : n'importe
-- quel ordre total strict convient.
variable (lt : Row n → Row n → Prop)

/-- Le predicat du noyau principal : `e < f o` (comparaison stricte -- cf. le
    masque `(1<<blo)-1` de la phase 3, qui exclut l'egalite). -/
def enum (p : Point n) : Prop := lt p.2 (f p.1)

/-- Le predicat du noyau diagonal : `e = f o`. -/
def diag (p : Point n) : Prop := p.2 = f p.1

/-- Un point diagonal est un point FIXE de la reflexion : il n'a pas de
    partenaire, d'ou son poids 1 au lieu de 2. -/
theorem diag_fixed {p : Point n} (hp : pinnedP p) (hd : diag p) : sigma p = p := by
  obtain ⟨o, e⟩ := p
  obtain ⟨h1, _⟩ := hp
  unfold diag at hd
  simp only at hd h1
  subst hd
  unfold sigma
  simp only [f_involutive h1]

/-- **Exactement un cas sur trois, et le troisieme est le miroir du premier.**

    Pour tout point epingle p, il arrive exactement une chose :
      * `enum p`  : p est enumere par `oe_kernel` (poids 2) ;
      * `diag p`  : p est son propre miroir, enumere par `diag_kernel` (poids 1) ;
      * `enum (sigma p)` : p n'est pas enumere, mais son miroir l'est.
    Les trois cas s'excluent deux a deux. -/
theorem reflexion_exact
    (tri : ∀ a b : Row n, lt a b ∨ a = b ∨ lt b a)
    (asym : ∀ a b : Row n, lt a b → ¬ lt b a)
    (irr : ∀ a : Row n, ¬ lt a a)
    {p : Point n} (hp : pinnedP p) :
    (enum lt p ∨ diag p ∨ enum lt (sigma p))
    ∧ ¬(enum lt p ∧ diag p)
    ∧ ¬(enum lt p ∧ enum lt (sigma p))
    ∧ ¬(diag p ∧ enum lt (sigma p)) := by
  obtain ⟨h1, h2⟩ := hp
  -- le predicat `enum` applique au miroir se lit `lt (f p.1) p.2`
  have hmir : enum lt (sigma p) ↔ lt (f p.1) p.2 := by
    unfold enum sigma
    simp only
    rw [f_involutive h2]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rcases tri p.2 (f p.1) with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (hmir.mpr h))
  · rintro ⟨he, hd⟩
    unfold enum at he; unfold diag at hd
    rw [hd] at he; exact irr _ he
  · rintro ⟨he, hs⟩
    exact asym _ _ he (hmir.mp hs)
  · rintro ⟨hd, hs⟩
    unfold diag at hd
    have := hmir.mp hs
    rw [hd] at this; exact irr _ this

/-- Le miroir d'un point epingle non enumere EST enumere : rien n'est perdu. -/
theorem sigma_rescues
    (tri : ∀ a b : Row n, lt a b ∨ a = b ∨ lt b a)
    (asym : ∀ a b : Row n, lt a b → ¬ lt b a)
    (irr : ∀ a : Row n, ¬ lt a a)
    {p : Point n} (hp : pinnedP p)
    (hne : ¬ enum lt p) (hnd : ¬ diag p) : enum lt (sigma p) := by
  rcases (reflexion_exact lt tri asym irr hp).1 with h | h | h
  · exact absurd h hne
  · exact absurd h hnd
  · exact h

end Reflexion

/- ------------------------------------------------------------------ -/
/- 3. Le theoreme de couverture                                        -/
/- ------------------------------------------------------------------ -/

section Couverture
variable (lt : Row n → Row n → Prop)

/-- **Couverture exacte.**  Tout point de l'espace -- epingle ou non -- possede
    un representant enumere, obtenu en le ramenant d'abord dans la partie
    epinglee par l'unique element du groupe de Klein qui convient, puis, si
    besoin, en lui appliquant la reflexion.  Le representant est enumere soit
    par `oe_kernel` (`enum`), soit par `diag_kernel` (`diag`). -/
theorem couverture_exacte
    (tri : ∀ a b : Row n, lt a b ∨ a = b ∨ lt b a)
    (asym : ∀ a b : Row n, lt a b → ¬ lt b a)
    (irr : ∀ a : Row n, ¬ lt a a)
    (p : Point n) :
    ∃ q : Point n, pinnedP q ∧ (enum lt q ∨ diag q) := by
  let k : Bool × Bool := (p.1 0, p.2 0)
  have hk : pinnedP (act k p) := (klein_exact p).1
  rcases (reflexion_exact lt tri asym irr hk).1 with h | h | h
  · exact ⟨act k p, hk, Or.inl h⟩
  · exact ⟨act k p, hk, Or.inr h⟩
  · exact ⟨sigma (act k p), sigma_pinned _, Or.inl h⟩

end Couverture

end Langford

/- ------------------------------------------------------------------ -/
/- Audit : aucun `sorry`, aucun axiome hors du noyau de Lean.          -/
/- ------------------------------------------------------------------ -/
#print axioms Langford.f_involutive
#print axioms Langford.sigma_involutive
#print axioms Langford.klein_exact
#print axioms Langford.klein_orbit_card_four
#print axioms Langford.diag_fixed
#print axioms Langford.reflexion_exact
#print axioms Langford.sigma_rescues
#print axioms Langford.couverture_exacte
