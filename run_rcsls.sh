#!/usr/bin/env sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

model=none # none or spectral

s=${1:-en}
t=${2:-ru}
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
python3 rcsls.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
  --lr 25 --niter 10 --model "${model}"

exit
echo "\nEvaluating---------------------------------------\n"
python3 reval.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
  --dico_test "${dico_test}"
