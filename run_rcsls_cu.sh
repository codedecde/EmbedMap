#!/usr/bin/env sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

model=none # none or spectral

GPUID=${1} # Pass GPU number as first arg
[ -z $GPUID ] && echo "Pass GPU ID as the first argument" && exit

s=${2:-en}
t=${3:-ru}
echo "${s}->${t} alignment"

if [ ! -d data/ ]; then
  mkdir -p data;
fi

if [ ! -d res/ ]; then
  mkdir -p res;
fi

dico_train=data/crosslingual/dictionaries/${s}-${t}.0-5000.txt
dico_test=data/crosslingual/dictionaries/${s}-${t}.5000-6500.txt
src_emb=data/wiki.${s}.vec
tgt_emb=data/wiki.${t}.vec

output=res/wiki.${s}-${t}.vec
echo "Model ${model}"

echo "\nAligning-----------------------------------------\n"
time python3 rcsls_cu.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
  --lr 25 --niter 10 --model "${model}" --gpu "${GPUID}"

exit
echo "\n\nEvaluating---------------------------------------\n"
python3 reval.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
  --dico_test "${dico_test}"
