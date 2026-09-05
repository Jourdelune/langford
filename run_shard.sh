#!/bin/sh
# Une tranche du calcul, pour execution distribuee (vast.ai spot, etc.).
#   ./run_shard.sh <index> <nb total de tranches> [n]
#
# Le travail par vhi n'est PAS uniforme : la reflexion fait que seuls les blocs
# d'indice >= vhi sont lances, donc la charge decroit lineairement en vhi. Un
# decoupage a vhi egaux donnerait des tranches allant du simple au vingtuple.
# On decoupe donc a TRAVAIL egal : la charge cumulee vaut v.NV - v^2/2, d'ou
#      v_i = NV . (1 - sqrt(1 - i/T))
#
# Sortie : une ligne PART=... a rassembler par
#      ./langford6 -n 31 --merge <toutes les PART> <celle de --diag>
# Une tranche interrompue est simplement rejouee ; rien d'autre a reprendre.
set -e
I=$1; T=$2; N=${3:-31}
NV=$(( 1 << ((N-1) - 7) ))
FROM=$(awk -v nv=$NV -v i=$I -v t=$T 'BEGIN{printf "%d", nv*(1-sqrt(1-i/t))}')
TO=$(awk   -v nv=$NV -v i=$I -v t=$T 'BEGIN{printf "%d", nv*(1-sqrt(1-(i+1)/t))}')
[ "$TO" -gt "$NV" ] && TO=$NV
CNT=$(( TO - FROM ))
[ "$CNT" -gt 0 ] || { echo "tranche vide" >&2; exit 0; }
echo "tranche $I/$T : vhi $FROM..$((TO-1))  ($CNT valeurs)" >&2

# --- tracabilite -------------------------------------------------------------
# Une somme partielle sans provenance n'est pas auditable : on ne sait ni quel
# binaire l'a produite, ni sur quelle carte, ni si toutes les taches ont tourne
# le meme code.  On emet donc, en plus du PART=, une ligne PROV= portant
# l'empreinte du binaire, la carte, le pilote et la duree.  `audit.sh` la relit.
SHA=$(sha256sum ./langford6 2>/dev/null | cut -c1-16)
# Le 4070 local tourne un binaire natif, les cartes louees le .fat
# (deux architectures, cudart statique) : deux empreintes differentes
# pour la MEME source.  C'est donc `src` qui doit etre unique, pas `sha`.
SRC=$(sha256sum ./langford6.cu 2>/dev/null | cut -c1-16)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
T0=$(date +%s)
OUT=$(./langford6 -n "$N" --from "$FROM" --count "$CNT" --chunk 64)
RC=$?
T1=$(date +%s)
echo "$OUT"
echo "PROV=task:$I/$T n:$N vhi:$FROM..$((TO-1)) src:${SRC:-?} sha:${SHA:-?} gpu:${GPU:-?} drv:${DRV:-?} sec:$((T1-T0)) utc:$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $RC
