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
    --d_model 8 \
    --d_ff 128 \
    --des 'Exp' \
    --itr 1 \
    --top_m 5 \
    --tau 1.0 \
    --alpha 0.95 \
    --dropout 0.5 \
    --batch_size 32 \
    --learning_rate 0.001 \
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
    --d_ff 256 \
    --des 'Exp' \
    --itr 1 \
    --top_m 4\
    --tau 10.0\
    --alpha 0.45\
    --batch_size 16\
    --dropout 0.4\
    --learning_rate 0.0001\
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
    --d_ff 256 \
    --top_k 5 \
    --top_m 4\
    --tau 0.05\
    --alpha 0.95\
    --batch_size 16\
    --dropout 0.4\
    --learning_rate 0.0001\
    --itr 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --d_model 8 \
    --d_ff 8 \
    --top_k 5 \
    --itr 1 \
    --top_m 3\
    --tau 0.05\
    --alpha 0.85\
    --dropout 0.4\
    --batch_size 64\
    --learning_rate 0.0003\
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
    --top_m 3\
    --tau 0.05\
    --alpha 0.55\
    --dropout 0.4\
    --batch_size 64\
    --learning_rate 0.0003\
    --itr 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --d_ff 256 \
    --top_k 5 \
    --top_m 4\
    --tau 10.0\
    --alpha 0.45\
    --dropout 0.4\
    --batch_size 16\
    --learning_rate 0.0001\
    --itr 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --d_ff 256 \
    --top_k 5 \
    --itr 1 \
    --top_m 4\
    --tau 10.0\
    --alpha 0.45\
    --batch_size 16\
    --dropout 0.4\
    --learning_rate 0.0001\
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
    --d_model 8 \
    --d_ff 256 \
    --top_k 5 \
    --itr 1 \
    --top_m 5\
    --tau 0.15\
    --alpha 0.95\
    --dropout 0.3\
    --batch_size 256\
    --learning_rate 0.0003\
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
    --d_model 256 \
    --d_ff 128 \
    --top_k 5 \
    --top_m 2 \
    --tau 1.0 \
    --alpha 0.85\
    --batch_size 8 \
    --dropout 0.4 \
    --learning_rate 0.001 \
    --itr 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 16 \
    --d_ff 256 \
    --batch_size 8 \
    --top_m 3 \
    --tau 5.0 \
    --alpha 0.85 \
    --dropout 0.3 \
    --learning_rate 0.0001 \
    --top_k 5 \
    --itr 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 8 \
    --d_ff 64 \
    --batch_size 16 \
    --top_m 3 \
    --tau 0.1 \
    --alpha 0.85 \
    --dropout 0.1 \
    --learning_rate 0.0003 \
    --top_k 5 \
    --train_epochs 1 \
    --itr 1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
