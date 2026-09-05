#!/bin/sh
# Verifie les valeurs de reference du README (§3.2c) contre un vrai calcul.
#   ./check_refs.sh          -> par le noyau GPU (rapide)
#   ./check_refs.sh ref      -> par l'algorithme classique (slice_ref, lent)
# Les tranches n=31 vhi=0 et vhi=4194304 coutent 15 et 8 min en mode `ref`.
MODE=${1:-gpu}
ok=0; ko=0
while read -r n v want; do
  case "$n" in \#*|"") continue;; esac
  if [ "$MODE" = ref ]; then got=$(./slice_ref "$n" "$v" | grep -oP 'PART=\K[0-9a-f:]+')
  else got=$(./langford6 -n "$n" --from "$v" --count 1 --chunk 1 2>/dev/null | grep -oP 'PART=\K[0-9a-f:]+'); fi
  if [ "$got" = "$want" ]; then printf "  n=%-3s vhi=%-9s OK\n" "$n" "$v"; ok=$((ok+1))
  else printf "  n=%-3s vhi=%-9s *** README=%s  calcul=%s\n" "$n" "$v" "$want" "$got"; ko=$((ko+1)); fi
done < refvals.txt
echo ""
echo "$ok valeur(s) de reference confirmee(s), $ko divergence(s)"
[ "$ko" -eq 0 ]
