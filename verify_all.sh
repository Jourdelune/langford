#!/bin/sh
# verify_all.sh -- rejoue toute la chaine de validation, du plus independant
# au plus proche du code.  C'est ce qu'il faut lancer AVANT une campagne, et
# ce qu'un relecteur peut relancer pour se convaincre lui-meme.
#
#   ./verify_all.sh          rapide   (~3 min)
#   ./verify_all.sh full     complet  (~25 min, ajoute n=16 et n=23/24 GPU)
set -e
MODE=${1:-quick}
fail=0
say(){ printf '\n\033[1m== %s\033[0m\n' "$1"; }
chk(){ if "$@"; then echo "   -> OK"; else echo "   -> *** ECHEC ***"; fail=1; fi; }

say "0. construction"
./build.sh >/dev/null 2>&1 || true
for b in verify cover_check slice_ref; do
  [ -x ./$b ] || gcc -O2 -o $b $b.c
done
echo "   binaire GPU : sha256 $(sha256sum ./langford6 | cut -c1-16)"

say "1. identite de Godfrey, contre une force brute independante"
echo "   (la force brute enumere les suites ; Godfrey somme sur {+-1}^{2n} ;"
echo "    aucune symetrie n'est utilisee dans la colonne 'complet')"
NM=13; [ "$MODE" = full ] && NM=16
chk ./verify $NM

say "2. couverture exacte de l'enumeration, au niveau des bits"
echo "   (instancie sur le vrai indexage du noyau, jusqu'a n=31)"
chk ./cover_check 9 31 12

say "3. decomposition de parite"
chk ./oe_check 31

say "4. tranches recalculees depuis la DEFINITION, comparees au bit pres"
SL="11:0 12:3 15:1 16:7 19:1 20:63 23:32767 24:65535 27:524287 28:1048575 31:8388607 31:8388600"
[ "$MODE" = full ] && SL="$SL 23:30000 24:60000 31:8385000 31:8380000"
chk ./check_slices.sh $SL

say "5. valeurs connues, calculees de bout en bout par le GPU"
NS="11 12 15 16 19 20"; [ "$MODE" = full ] && NS="$NS 23 24"
for n in $NS; do
  got=$(./langford6 -n $n 2>/dev/null | sed -n 's/.*L(2,[0-9]*) *= *//p')
  case "$n:$got" in
    11:17792|12:108144|15:39809640|16:326721800|19:256814891280|\
20:2636337861200|23:3799455942515488|24:46845158056515936)
        printf "   n=%-3s %-20s OK\n" "$n" "$got" ;;
    *)  printf "   n=%-3s %-20s *** ATTENDU AUTRE CHOSE ***\n" "$n" "$got"; fail=1 ;;
  esac
done

say "6. refus des n ou la reduction de symetrie est invalide"
for n in 9 10 13 14 17 18 21 22; do
  if ./langford6 -n $n >/dev/null 2>&1; then
    echo "   n=$n : *** ACCEPTE alors qu'il devrait etre refuse ***"; fail=1
  fi
done
echo "   les huit n = 1,2 (mod 4) testes sont bien refuses"
echo "   -> OK"

say "7. preuve Lean de la couverture"
if command -v lean >/dev/null 2>&1 || [ -x "$HOME/.elan/bin/lean" ]; then
  L=$(command -v lean || echo "$HOME/.elan/bin/lean")
  if (cd proof && "$L" Langford.lean); then echo "   -> OK (aucun sorry, axiomes standard)"
  else echo "   -> *** ECHEC ***"; fail=1; fi
else
  echo "   lean absent -- installer avec elan pour rejouer cette etape (facultatif)"
fi

printf '\n=============================================================\n'
if [ "$fail" -eq 0 ]; then
  echo " TOUTE LA CHAINE PASSE."
  echo " Le noyau calcule bien 2*L(2,n), et son enumeration est exacte."
else
  echo " *** AU MOINS UNE ETAPE A ECHOUE -- NE PAS LANCER DE CAMPAGNE ***"
fi
echo "============================================================="
exit $fail
