cd /home/lc1526469/time_series/ICML/KUP-BI/TimesNet
export CUDA_VISIBLE_DEVICES=2
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=TimesNet
model_id_name=TimesNet

for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTh1.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTh1 \
    --top_m 9 \
    --tau 1.0 \
    --alpha 0.35 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 3 \
    --top_k 5 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTh2.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTh2 \
    --top_m 5 \
    --tau 1.0 \
    --alpha 0.35 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 3 \
    --top_k 5 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 96 192 
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTm1.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm1 \
    --top_m 7 \
    --tau 10.0 \
    --alpha 0.9 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --itr 3 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done



for pred_len in 336
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTm1.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm1 \
    --top_m 9 \
    --tau 0.15 \
    --alpha 0.9 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 16 \
    --d_ff 32 \
    --top_k 5 \
    --itr 3 \
    --train_epochs 3 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTm1.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm1 \
    --top_m 5 \
    --tau 1.0 \
    --alpha 0.9 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 16 \
    --d_ff 32 \
    --top_k 5 \
    --itr 3 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 96 336
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTm2.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm2 \
    --top_m 9 \
    --tau 0.15 \
    --alpha 0.35 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --itr 3 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 192
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTm2.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm2 \
    --top_m 5 \
    --tau 0.1 \
    --alpha 0.65 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --itr 3 \
    --train_epochs 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96/ETT-small \
    --data_path ETTm2.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data ETTm2 \
    --top_m 9 \
    --tau 0.1 \
    --alpha 0.9 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 16 \
    --d_ff 32 \
    --top_k 5 \
    --itr 3 \
    --train_epochs 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 24 36 48 60
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96 \
    --data_path national_illness.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data custom \
    --top_m 3 \
    --tau 0.05 \
    --alpha 0.45 \
    --features M \
    --seq_len 36 \
    --pred_len $pred_len \
    --label_len 18 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 768 \
    --d_ff 768 \
    --top_k 5 \
    --itr 3 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96 \
    --data_path exchange_rate.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data custom \
    --features M \
    --top_m 7 \
    --tau 0.15 \
    --alpha 0.45 \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --itr 3 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_96 \
    --data_path exchange_rate.csv \
    --model_id $model_id_name \
    --model $model_name \
    --data custom \
    --features M \
    --top_m 3 \
    --tau 5.0 \
    --alpha 0.55 \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --train_epochs 1\
    --itr 3 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
