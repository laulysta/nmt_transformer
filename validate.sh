#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
#nematus=/home/sebastien/Documents/Git/nematus

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/home/sebastien/Documents/Git/mosesdecoder

# theano device, in case you do not want to compute on gpu, change it to cpu
#device=cpu

#model prefix
#prefix=model/model.npz

#dev=data/newsdev2016.bpe.ro
#ref=data/newsdev2016.tok.en

model_name=$1
dev=$2
ref=$3
data=$4

# decode
#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
#     -m $prefix.dev.npz \
#     -i $dev \
#     -o $dev.output.dev \
#     -k 1 -n -p 5
python3 translate.py \
-model $model_name \
-src $dev \
-vocab $data \
-output ${model_name}.output.dev \
-beam_size 1 \
-no_cuda


sed -r 's/(@@ )|(@@ ?$)//g' < ${model_name}.output.dev > ${model_name}.output.dev.tmp
mv ${model_name}.output.dev.tmp ${model_name}.output.dev

## get BLEU
#BEST=`cat ${model_name}_best_bleu || echo 0`
./multi-bleu.perl $ref < ${model_name}.output.dev >> ${model_name}_bleu_scores
BLEU=`./multi-bleu.perl $ref < ${model_name}.output.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
#BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# # save model with highest BLEU
# if [ "$BETTER" = "1" ]; then
#   echo "new best; saving"
#   echo $BLEU > ${prefix}_best_bleu
#   cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
# fi
