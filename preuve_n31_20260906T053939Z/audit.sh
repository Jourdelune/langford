#!/bin/sh
# audit.sh -- de quoi faire relire le resultat par quelqu'un d'autre.
#
#   ./audit.sh <n> <nb taches> [fichier parts]
#
# Un total juste ne prouve rien tout seul : il faut pouvoir dire QUI a calcule
# QUOI, avec QUEL binaire, et montrer qu'un recalcul independant retombe sur les
# memes bits.  Ce script produit ce dossier :
#
#   1. completude et unicite des taches (ce que fait deja collect.sh) ;
#   2. divisibilite par 2^{2n+1} du total -- un test qu'une seule tranche
#      fausse fait echouer ;
#   3. inventaire des binaires : toutes les taches doivent porter la MEME
#      empreinte sha256, sinon on melange deux codes ;
#   4. inventaire des cartes et des pilotes ;
#   5. RECALCUL REDONDANT : un echantillon aleatoire de taches est refait ici,
#      et compare au bit pres a ce qu'a rendu le worker distant.
set -e
N=${1:-31}; T=${2:-4096}; F=${3:-parts_n${N}.txt}
SAMPLE=${SAMPLE:-5}
[ -f "$F" ] || { echo "fichier absent : $F" >&2; exit 1; }

echo "=============================================================="
echo " DOSSIER D'AUDIT   n=$N   $T taches   fichier $F"
echo " genere le $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

echo ""
echo "--- 1. completude et unicite -------------------------------"
DUP=$(awk '{print $1}' "$F" | sort | uniq -d)
[ -z "$DUP" ] && echo "  aucune tache en double" || { echo "  *** doublons : $DUP"; exit 1; }
MISS=$(for k in $(seq 0 $((T-1))); do grep -q "^#$k " "$F" || echo $k; done | tr '\n' ' ')
grep -q '^#D ' "$F" || MISS="$MISS diagonale(#D)"
[ -z "$(echo $MISS)" ] && echo "  les $T taches et la diagonale sont presentes" \
                       || { echo "  *** manquantes : $MISS"; exit 1; }

echo ""
echo "--- 2. auto-test arithmetique sur le total -----------------"
./langford6 -n "$N" --merge $(awk '{print $2}' "$F") | sed 's/^/  /'

echo ""
echo "--- 3. code utilise ----------------------------------------"
echo "  SOURCE (doit etre unique -- c'est le seul invariant qui compte) :"
grep -o 'src:[0-9a-f?]*' "$F" | sort | uniq -c | sed 's/^/    /'
NSRC=$(grep -o 'src:[0-9a-f?]*' "$F" | sort -u | wc -l)
if [ "$NSRC" -le 1 ]; then echo "    -> une seule source : bien"
else echo "    -> *** $NSRC sources differentes : le run melange deux codes ***"; fi
echo "  binaires (plusieurs sont NORMAUX : le 4070 local compile en natif,"
echo "  les cartes louees tournent le .fat -- meme source, builds differents) :"
grep -o 'sha:[0-9a-f?]*' "$F" | sort | uniq -c | sed 's/^/    /'
echo "  source locale actuelle : sha256 $(sha256sum ./langford6.cu | cut -c1-16)"

echo ""
echo "--- 4. cartes et pilotes -----------------------------------"
grep -o 'gpu:[^ ,]*' "$F" | sort | uniq -c | sed 's/^/  /'
grep -o 'drv:[^ ,]*' "$F" | sort | uniq -c | sed 's/^/  /'

echo ""
echo "--- 5. recalcul redondant d'un echantillon -----------------"
echo "  ($SAMPLE taches retirees au hasard et refaites ici)"
KS=$(awk '$1 ~ /^#[0-9]+$/ {print substr($1,2)}' "$F" | shuf -n "$SAMPLE" 2>/dev/null || \
     awk '$1 ~ /^#[0-9]+$/ {print substr($1,2)}' "$F" | head -n "$SAMPLE")
bad=0
for k in $KS; do
  want=$(awk -v k="#$k" '$1==k{print $2}' "$F")
  got=$(./run_shard.sh "$k" "$T" "$N" 2>/dev/null | sed -n 's/^PART=\([0-9a-f:]*\).*/\1/p')
  if [ "$want" = "$got" ]; then printf "  tache %-6s IDENTIQUE\n" "$k"
  else printf "  tache %-6s *** DIVERGENCE ***  fichier=%s  recalcul=%s\n" "$k" "$want" "$got"; bad=1; fi
done
[ "$bad" -eq 0 ] && echo "  -> l'echantillon se reproduit au bit pres" \
                 || { echo "  -> *** AU MOINS UNE TACHE NE SE REPRODUIT PAS ***"; exit 1; }

echo ""
echo "=============================================================="
echo " Reproduire cet audit :  ./audit.sh $N $T $F"
echo " Verifier la chaine complete :  ./verify_all.sh"
echo "=============================================================="
