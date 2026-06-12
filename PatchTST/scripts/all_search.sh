cd /home/lc1526469/time_series/ICML/KUP-BI/PatchTST
export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=PatchTST
model_id_name=PatchTST
random_seed=2021

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
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
      --e_layers 2 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 512 \
      --dropout 0.5\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --top_m 6\
      --tau 1.0\
      --alpha 0.9\
      --itr 1 --batch_size 128 --learning_rate 0.0001 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
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
      --e_layers 2 \
      --n_heads 8 \
      --d_model 32 \
      --d_ff 64 \
      --dropout 0.5\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --top_m 8\
      --tau 10.0\
      --alpha 0.95\
      --itr 1 --batch_size 128 --learning_rate 0.0003 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
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
      --e_layers 4 \
      --n_heads 32 \
      --d_model 256 \
      --d_ff 256 \
      --dropout 0.4\
      --fc_dropout 0.5\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4\
      --top_m 1\
      --tau 1.0\
      --alpha 0.95\
      --itr 1 --batch_size 128 --learning_rate 0.0003 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336/ETT-small \
      --data_path ETTm2.csv \
      --model_id $model_id_name\
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 64 \
      --dropout 0.5\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --top_m 5\
      --tau 1.0\
      --alpha 0.95\
      --itr 1 --batch_size 128 --learning_rate 0.0003 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done



for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336 \
      --data_path national_illness.csv \
      --model_id $model_id_name \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 104 \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 4 \
      --n_heads 4 \
      --d_model 64 \
      --d_ff 18 \
      --dropout 0.4\
      --fc_dropout 0.5\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'constant'\
      --top_m 10\
      --tau 0.1\
      --alpha 0.95\
      --itr 1 --batch_size 16 --learning_rate 0.0003 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path /home/lc1526469/time_series/ICML/KUP-BI/dataset_336 \
      --data_path exchange_rate.csv \
      --model_id $model_id_name \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 4 \
      --n_heads 32\
      --d_model 512 \
      --d_ff 16\
      --dropout 0.1\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --top_m 8\
      --tau 0.01\
      --alpha 0.2\
      --itr 1 --batch_size 16 --learning_rate 0.0003 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


