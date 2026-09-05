#!/bin/sh
# Execute un lot de taches et emet une ligne "<tache> <somme partielle>" par
# tache TERMINEE, au fil de l'eau.  C'est le maitre qui decide du lot ; ici on
# se contente de calculer et d'emettre immediatement, pour qu'une coupure de
# connexion ne fasse perdre que la tache en cours et pas tout le lot.
#   ./run_batch.sh <n> <T> <tache...>        (tache "D" = orbites fixes)
N=$1; T=$2; shift 2
# Provenance : constante pour toute la machine, calculee une fois.  Elle est
# emise en TROISIEME champ de chaque ligne -- le maitre la range en base, et
# audit.sh la relit pour verifier que toutes les taches ont tourne le meme
# binaire sur une carte identifiee.  Sans elle, une somme partielle n'est pas
# auditable : on ne sait ni quel code l'a produite, ni sur quoi.
SHA=$(sha256sum ./langford6 2>/dev/null | cut -c1-16)
# Le 4070 local tourne un binaire natif, les cartes louees le .fat
# (deux architectures, cudart statique) : deux empreintes differentes
# pour la MEME source.  C'est donc `src` qui doit etre unique, pas `sha`.
SRC=$(sha256sum ./langford6.cu 2>/dev/null | cut -c1-16)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
PROV="src:${SRC:-?},sha:${SHA:-?},gpu:${GPU:-?},drv:${DRV:-?}"
for k in "$@"; do
  T0=$(date +%s)
  if [ "$k" = "D" ]; then
    R=$(./langford6 -n "$N" --diag 2>/dev/null | sed -n 's/^PART=\([0-9a-f:]*\).*/\1/p')
  else
    R=$(./run_shard.sh "$k" "$T" "$N" 2>/dev/null | sed -n 's/^PART=\([0-9a-f:]*\).*/\1/p')
  fi
  if [ -n "$R" ]; then printf '%s %s %s,sec:%s,utc:%s\n' "$k" "$R" "$PROV" \
       "$(( $(date +%s) - T0 ))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else printf 'ERR %s\n' "$k"; fi
done
