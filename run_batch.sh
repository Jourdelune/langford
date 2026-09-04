#!/bin/sh
# Execute un lot de taches et emet une ligne "<tache> <somme partielle>" par
# tache TERMINEE, au fil de l'eau.  C'est le maitre qui decide du lot ; ici on
# se contente de calculer et d'emettre immediatement, pour qu'une coupure de
# connexion ne fasse perdre que la tache en cours et pas tout le lot.
#   ./run_batch.sh <n> <T> <tache...>        (tache "D" = orbites fixes)
N=$1; T=$2; shift 2
for k in "$@"; do
  if [ "$k" = "D" ]; then
    R=$(./langford6 -n "$N" --diag 2>/dev/null | sed -n 's/^PART=\([0-9a-f:]*\).*/\1/p')
  else
    R=$(./run_shard.sh "$k" "$T" "$N" 2>/dev/null | sed -n 's/^PART=\([0-9a-f:]*\).*/\1/p')
  fi
  if [ -n "$R" ]; then printf '%s %s\n' "$k" "$R"
  else printf 'ERR %s\n' "$k"; fi
done
