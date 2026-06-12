cd /home/lc1526469/time_series/ICML/KUP-BI/xPatch
export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=xPatch
model_id_name=xPatch
seq_len=512
ma_type=ema
alpha=0.3
beta=0.3

for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_512/ETT-small \
      --data_path ETTh2.csv \
      --model_id $model_id_name \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.0001 \
      --lradj 'type3'\
      --train_epochs 20 \
      --patience 5 \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --top_m 3 \
      --tau 1.0 \
      --CalphaC 0.95 \
      --use_dynamic_gate 1
done
