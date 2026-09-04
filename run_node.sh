#!/bin/sh
# Un noeud multi-GPU : une carte = un worker.   ./run_node.sh [n] [nb taches]
#
# Sans --dev, les processus tapent TOUS sur le device 0 et les autres cartes
# restent inutilisees -- sur un noeud 8x facture a l'heure, c'est 7/8 du loyer
# jete. On repartit donc explicitement.
#
# Chaque worker n'ecrit que ses propres taches (i, i+G, i+2G, ...), donc aucun
# conflit sur parts_n<N>.txt : sous Linux un append de moins de 4 Ko est
# atomique, et deux workers ne visent jamais la meme tache.
set -e
N=${1:-31}; T=${2:-4096}
G=$(nvidia-smi --list-gpus | wc -l)
[ "$G" -gt 0 ] || { echo "aucun GPU detecte" >&2; exit 1; }
echo "$G GPU detectes, $T taches, n=$N" >&2
i=0
while [ "$i" -lt "$G" ]; do
  CUDA_VISIBLE_DEVICES=$i ./worker.sh "$i" "$G" "$N" "$T" > log_gpu$i.txt 2>&1 &
  i=$((i+1))
done
wait
echo "noeud termine ; ./collect.sh $N $T" >&2
