# XiYan-SQL baseline: GPU run (round v3)

Runs `XGenerationLab/XiYanSQL-QwenCoder-14B-2504` in bf16 with the prompt of
`eval_xiyan_duck.py` (unchanged) on the six datasets, three seeds each, and saves
only the generated queries. The metrics are computed later, on the Norma side,
by `experiments/baselines/evaluate_baselines_v3.py xiyan`.

## Machine

- 1 GPU with at least 40 GB (A100 80 GB recommended, as in the paper), CUDA 12.x
- about 60 GB of disk (model about 30 GB, data and outputs)
- 32 GB of RAM (the HMDA slices are loaded with pandas to build the schema)

## Steps

```bash
# 0. on RunPod: work on the persistent volume and keep the model cache there
cd /workspace && tar -xzf baselines_v3_gpu_package.tar.gz && cd baselines_v3_gpu_package
export HF_HOME=/workspace/hf_cache
tmux new -s xiyan          # reattach later with: tmux attach -t xiyan

# 1. environment
python3 -m venv venv && source venv/bin/activate
pip install -U pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6"
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. run (the model is downloaded on the first call; no token needed)
bash run_all.sh

# 3. bring back outputs.tar.gz
```

Expected time: under 1.5 h for all six datasets and seeds (about 1,500 generations).
If a dataset fails, the log in `logs/<dataset>.log` says why; it can be rerun alone with
the corresponding line of `run_all.sh`.

## Variant B (prompt of xyyan_text2sql_inference.py, XiYan-SQL 32B)

Same steps, with a larger volume (150 GB: the 32B model takes about 65 GB) and an
A100 80 GB; then run `bash run_all_B.sh` instead of `run_all.sh` and bring back
`outputs_B.tar.gz`. Expected time: about 2-2.5 h.
