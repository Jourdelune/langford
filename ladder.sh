#!/bin/sh
# ladder.sh -- l'echelle complete de n=1 a n=24, et comment chaque n est couvert.
#
# Trois regimes :
#   n = 1..16          force brute + Godfrey complet + Klein x4  (./verify)
#   n = 1,2 (mod 4)    aucune suite n'existe ; le noyau doit REFUSER
#   n = 19,20,23,24    valeur connue, calculee de bout en bout par le GPU
set -e
GPU=${GPU:-1}
echo "  n | attendu L(2,n)          | couverture"
echo "----+-------------------------+--------------------------------------------"
val(){ case $1 in
  1|2|5|6|9|10|13|14|17|18|21|22) echo 0 ;;
  3) echo 1;; 4) echo 1;; 7) echo 26;; 8) echo 150;;
  11) echo 17792;; 12) echo 108144;; 15) echo 39809640;; 16) echo 326721800;;
  19) echo 256814891280;; 20) echo 2636337861200;;
  23) echo 3799455942515488;; 24) echo 46845158056515936;; esac; }
fail=0
for n in $(seq 1 24); do
  V=$(val $n); how=""
  if [ "$n" -le 16 ]; then how="force brute + Godfrey complet (./verify 16)"; fi
  case $((n % 4)) in
    1|2) if ./langford6 -n $n >/dev/null 2>&1; then how="$how  *** NOYAU ACCEPTE A TORT ***"; fail=1
         else how="${how:+$how ; }noyau refuse (reduction x4 invalide)"; fi ;;
    *)   if [ "$n" -ge 11 ] && [ "$GPU" = 1 ]; then
           g=$(./langford6 -n $n 2>/dev/null | sed -n 's/.*L(2,[0-9]*) *= *//p')
           if [ "$g" = "$V" ]; then how="${how:+$how ; }GPU bout en bout OK"
           else how="${how:+$how ; }*** GPU a rendu $g ***"; fail=1; fi
         elif [ "$n" -lt 9 ]; then how="${how:+$how ; }hors portee du decoupage GPU (n>=9)"
         fi ;;
  esac
  printf "%3d | %23s | %s\n" "$n" "$V" "$how"
done
echo ""
[ "$fail" -eq 0 ] && echo "echelle 1..24 : tout concorde" || { echo "*** DIVERGENCE ***"; exit 1; }
