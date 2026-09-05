#!/bin/sh
# Un worker sur une machine louee.   ./worker.sh <i> <nb workers> [n] [nb taches]
#
# Le travail total est decoupe en T taches a charge EGALE (cf. run_shard.sh).
# Le worker i traite les taches i, i+W, i+2W, ...  Chaque tache resolue est
# ajoutee a parts_n<N>.txt sous la forme "#<tache> <somme partielle>".
#
# Reprise apres preemption : relancer la meme commande.  Les taches deja
# presentes dans le fichier sont sautees, donc une instance spot tuee ne coute
# qu'une tache (~3 min si T est bien choisi).  Rien a synchroniser entre
# workers : aucune communication, sortie de 40 octets par tache.
set -e
I=$1; W=$2; N=${3:-31}; T=${4:-4000}
[ -n "$W" ] || { echo "usage: $0 <i> <nb_workers> [n] [nb_taches]" >&2; exit 1; }
OUT=parts_n${N}.txt
: >> $OUT
if [ "$I" -eq 0 ] && ! grep -q '^#D ' $OUT; then      # orbites fixes : une fois
  D=$(./langford6 -n $N --diag 2>/dev/null | sed -n 's/^PART=\([0-9a-f:]*\).*/\1/p')
  [ -n "$D" ] && echo "#D $D" >> $OUT
fi
K=$I
while [ "$K" -lt "$T" ]; do
  if ! grep -q "^#$K " $OUT; then
    O=$(./run_shard.sh "$K" "$T" "$N" 2>/dev/null)
    R=$(echo "$O" | sed -n 's/^PART=\([0-9a-f:]*\).*/\1/p')
    P=$(echo "$O" | sed -n 's/^PROV=//p')
    # Champs 1 et 2 inchanges (#tache, somme) : collect.sh continue de marcher.
    # La provenance est appendue derriere, pour audit.sh et pour un relecteur.
    if [ -n "$R" ]; then echo "#$K $R $P" >> $OUT
    else echo "tache $K : aucune sortie" >&2; exit 1; fi
    echo "worker $I : tache $K/$T faite  ($(grep -c '^#' $OUT) au total)" >&2
  fi
  K=$((K+W))
done
echo "worker $I termine" >&2
