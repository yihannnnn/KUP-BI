cd /home/lc1526469/time_series/ICML/KUP-BI/xPatch
export CUDA_VISIBLE_DEVICES=0

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
      --data_path ETTh1.csv \
      --model_id $model_id_name \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0005 \
      --lradj 'type3'\
      --train_epochs 20 \
      --patience 5 \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --top_m 7 \
      --tau 0.01 \
      --CalphaC 0.45 
done

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


for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_512/ETT-small \
      --data_path ETTm1.csv \
      --model_id $model_id_name \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 256 \
      --learning_rate 0.0001 \
      --lradj 'type3'\
      --train_epochs 20 \
      --patience 5 \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --top_m 1 \
      --tau 0.01 \
      --CalphaC 1.0 
done


for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_512/ETT-small \
      --data_path ETTm2.csv \
      --model_id $model_id_name \
      --model $model_name \
      --data ETTm2\
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 256 \
      --learning_rate 0.0005 \
      --lradj 'type3'\
      --train_epochs 20 \
      --patience 5 \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --top_m 5 \
      --tau 0.01 \
      --CalphaC 0.95 
done



for pred_len in 24 36 48 60
do
  python -u run.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_512 \
    --data_path national_illness.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 104 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lradj 'type1'\
    --patch_len 8 \
    --stride 4 \
    --train_epochs 10 \
    --patience 3 \
    --ma_type 'reg' \
    --alpha $alpha \
    --beta $beta \
    --top_m 5 \
    --tau 0.05 \
    --CalphaC 0.95
done



for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_512 \
    --data_path exchange_rate.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj 'type1'\
    --train_epochs 10 \
    --patience 3 \
    --ma_type 'reg' \
    --alpha $alpha \
    --beta $beta \
    --top_m 9 \
    --tau 10.0 \
    --CalphaC 0.75 
done
