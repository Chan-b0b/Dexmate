# Remote Training Setup (H200)

## First-time Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-org/CNS_code.git
cd CNS_code/LGES/vla_training
```

### 2. Create virtual environment
```bash
python3 -m venv vla_venv
source vla_venv/bin/activate
```

### 3. Install PyTorch for H200 (CUDA 12.1+)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Login to HuggingFace (optional, for private datasets)
```bash
huggingface-cli login
```
When prompted, paste your HuggingFace access token.

## Running Training

### Fresh training run
```bash
source vla_venv/bin/activate
RUN_NAME=h200_run_01 ./train_smolvla.sh --steps=60000
```

The script will:
- Automatically download the dataset from `chanho-lee/lges_suction` on HuggingFace
- Cache it in `~/.cache/huggingface/datasets/`
- Save outputs to `outputs/h200_run_01/`
- Log to `logs/h200_run_01/`

### Resume training
```bash
RUN_NAME=h200_run_01 ./train_smolvla.sh --resume --steps=80000
```

### Use different HuggingFace dataset
```bash
HF_DATASET_REPO=chanho-lee/custom_dataset RUN_NAME=test ./train_smolvla.sh
```

## Monitoring Training

### View TensorBoard
```bash
source vla_venv/bin/activate
tensorboard --logdir logs/h200_run_01/tb
```
Then navigate to `http://<remote-server>:6006` in your browser.

### View training log
```bash
tail -f logs/h200_run_01/train.log
```

## Keep Code Updated
```bash
git pull origin Split_Episode
```

## Troubleshooting

**Dataset download fails:**
- Check HuggingFace token: `huggingface-cli login`
- Check internet connectivity to HuggingFace

**CUDA out of memory:**
- Reduce `--batch_size` in the script (default is 32)
- Increase `--log_freq` to log less frequently

**Slow training:**
- Check GPU utilization: `nvidia-smi` (should be near 100%)
- Increase `--num_workers` (currently 12) if CPU is idle
