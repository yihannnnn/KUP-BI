## Contact

If you have any questions, please feel free to contact us at yihann@stu.scu.edu.cn.

# KUP-BI

Official implementation for the paper:

**Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting**

KUP-BI is a plug-in paradigm for time series forecasting. It constructs a train-only historical library from history-target-post-target continuation chains, estimates continuation-style auxiliary inputs through retrieval-based ratio aggregation, and fuses the auxiliary stream with forecasting backbones.

## Models

This repository includes KUP-BI implementations based on:

- DLinear
- PatchTST
- TimesNet
- xPatch

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the benchmark datasets and place them under `dataset/`.

The datasets can be obtained from the dataset links provided by the iTransformer repository:

Google Drive dataset archive: https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link

Expected examples:

```text
dataset/
  ETT-small/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
  national_illness.csv
  exchange_rate.csv
```

## Usage

Each model folder contains scripts for running experiments.

### DLinear

```bash
cd DLinear
sh scripts/kup-bi_search.sh
```

### PatchTST

```bash
cd PatchTST
sh scripts/kup-bi_search.sh
```

### TimesNet

```bash
cd TimesNet
sh scripts/kup-bi_search.sh
```

### xPatch

```bash
cd xPatch
sh scripts/kup-bi_search.sh
```

Before running the scripts, please modify the dataset paths in the scripts according to your local environment.

## Example

For example, to run KUP-BI with xPatch on ETTh1:

```bash
cd xPatch

python run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --model_id xPatch \
  --model xPatch \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 7 \
  --des Exp \
  --itr 1
```

## Main Idea

KUP-BI introduces a continuation-style auxiliary stream for time series forecasting.

Given a training chain consisting of:

```text
history -> target -> post-target continuation
```

KUP-BI builds a train-only historical library and estimates an auxiliary continuation-style input for each query window. The auxiliary input is then fused with the original input through a lightweight gated fusion module.

The framework can be plugged into different forecasting backbones without changing the main forecasting objective.

## Acknowledgement

We appreciate the following GitHub repositories for their valuable code and efforts:

- PatchTST: https://github.com/yuqinie98/PatchTST
- xPatch: https://github.com/stitsyuk/xPatch
- DLinear: https://github.com/cure-lab/LTSF-Linear
- TimesNet: https://github.com/thuml/Time-Series-Library

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{liu2026kupbi,
  title={Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting},
  author={Liu, Chong and Zhou, Yingjie and Li, Hao and Wang, Pengyang and Wen, Qingsong and Zhu, Ce},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
