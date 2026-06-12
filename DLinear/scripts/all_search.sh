cd /home/lc1526469/time_series/ICML/KUP-BI/DLinear
export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=DLinear
model_id_name=DLinear

for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336/ETT-small \
    --data_path ETTh1.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --top_m 1 \
    --tau 0.15 \
    --alpha 0.95 \
    --itr 1 --batch_size 128 --learning_rate 0.03 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336/ETT-small \
    --data_path ETTh2.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --top_m 1 \
    --tau 0.05 \
    --alpha 0.15 \
    --itr 1 --batch_size 32 --learning_rate 0.005 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336/ETT-small \
    --data_path ETTm1.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --top_m 10 \
    --tau 10.0 \
    --alpha 0.95 \
    --itr 1 --batch_size 256 --learning_rate 0.001 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336/ETT-small \
    --data_path ETTm2.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --top_m 10 \
    --tau 0.01 \
    --alpha 0.8 \
    --itr 1 --batch_size 16 --learning_rate 0.001 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done



for pred_len in 24 36 48 60
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336 \
    --data_path national_illness.csv  \
    --model_id $model_id_name \
    --model $model_name \
    --data custom \
    --features M \
    --label_len 18 \
    --seq_len 104 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --top_m 4\
    --tau 10.0\
    --alpha 0.95\
    --itr 1 --batch_size 8 --learning_rate 0.05 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336 \
    --data_path exchange_rate.csv  \
    --model_id $model_id_name \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --top_m 3\
    --tau 0.05\
    --alpha 0.25\
    --itr 1 --batch_size 32 --learning_rate 0.01 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done