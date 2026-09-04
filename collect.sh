#!/bin/sh
# Rassemble les sommes partielles de tous les workers et conclut.
#   ./collect.sh [n] [nb taches] [fichiers...]
# Verifie d'abord qu'aucune tache ne manque et qu'aucune n'est en double, puis
# laisse l'auto-test de --merge (divisibilite par 2^2n) valider l'ensemble.
set -e
N=${1:-31}; T=${2:-4000}; shift 2 2>/dev/null || true
FILES=${*:-parts_n${N}.txt}
ALL=$(cat $FILES)
NB=$(echo "$ALL" | grep -c '^#' || true)
DUP=$(echo "$ALL" | awk '{print $1}' | sort | uniq -d)
[ -z "$DUP" ] || { echo "*** taches en double : $DUP" >&2; exit 1; }
MISS=$(for k in $(seq 0 $((T-1))); do echo "$ALL" | grep -q "^#$k " || echo $k; done | tr '\n' ' ')
echo "$ALL" | grep -q '^#D ' || MISS="$MISS diagonale(#D)"
[ -z "$(echo $MISS)" ] || { echo "*** taches manquantes : $MISS" >&2; exit 1; }
echo "$NB / $((T+1)) taches presentes, aucune en double." >&2
exec ./langford6 -n "$N" --merge $(echo "$ALL" | awk '{print $2}')
