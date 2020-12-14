#!/bin/bash
arch=tagtransformer

data_dir=cmudict
dataset=cmudict
ckpt_dir=checkpoints/transformer

#--train $data_dir/$dataset.trn \
python src/decode.py \
    --dataset infp2word \
    --trgt_vocab $data_dir/trgt.txt --src_vocab $data_dir/source.txt \
    --test $data_dir/input_example.tsv \
    --model $ckpt_dir/default/$dataset --load smart \
    --arch $arch --gpuid 0 --bestacc \
    --decode greedy
