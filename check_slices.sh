#!/bin/sh
# Compare, tranche par tranche et AU BIT PRES, le noyau GPU contre slice_ref,
# qui recalcule la meme somme a partir de la definition de Godfrey sans rien
# partager avec lui (ni parite, ni SWAR, ni PTX, ni chemin rapide).
ok=0; ko=0
for spec in "$@"; do
  n=${spec%%:*}; v=${spec##*:}
  a=$(./slice_ref  "$n" "$v"                                  | grep -oP 'PART=\K[0-9a-f:]+')
  b=$(./langford6 -n "$n" --from "$v" --count 1 --chunk 1 2>/dev/null | grep -oP 'PART=\K[0-9a-f:]+')
  if [ -n "$a" ] && [ "$a" = "$b" ]; then
    printf "  n=%-3s vhi=%-9s IDENTIQUE  %s\n" "$n" "$v" "$a"; ok=$((ok+1))
  else
    printf "  n=%-3s vhi=%-9s *** DIVERGENCE ***  ref=%s  gpu=%s\n" "$n" "$v" "$a" "$b"; ko=$((ko+1))
  fi
done
echo ""
echo "$ok tranche(s) identiques au bit pres, $ko divergence(s)"
[ "$ko" -eq 0 ]
