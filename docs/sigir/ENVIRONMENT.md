# Environment of the registered runs (collected 2026-10-05; no credential, no host name)

| item | value |
|---|---|
| GPU | one NVIDIA GeForce RTX 4090 D, 24,564 MiB, driver 595.71.05 |
| CPU / RAM | Intel Xeon Platinum 8352S, 128 logical CPUs, 503 GiB RAM |
| OS | Linux 5.15.0-94-generic x86_64 |
| Python | 3.11.16 (conda environment `lumen`) |
| torch | 2.13.0+cu130 (CUDA 13.0, cuDNN 9.2) |
| vLLM | 0.30.0 (fp16 prefill-only scoring, top-50 log-probabilities, Yes/No read by token id) |
| transformers / peft / accelerate / tokenizers / safetensors | 5.18.0 / 0.21.2 / 1.15.0 / 0.23.2 / 0.8.0 |
| numpy / scipy / pandas / scikit-learn | 2.3.5 / 1.17.1 / 3.0.6 / 1.9.1 |
| backbones | Qwen3-8B (5 safetensors shards, 16 GB) and Llama-3.1-8B-Instruct (4 shards, 30 GB with the `original` directory), local copies; file checksums in `docs/sigir/MODEL_SHA256.txt` |
| fine-tuning | LoRA r 16, alpha 32, dropout 0.05 on q/k/v/o, lr 1e-4, cosine, 3% warm-up, 1 epoch (FT-L: 3 epochs), bf16, effective batch 32 |

Nondeterminism: the same prompts rescored by vLLM can differ in the fourth decimal of a mean UAUC because batching changes the floating-point summation order
(example: two scorings of the same ML-1M zero-shot prompts, that of the grid report and that of the Gate-FT file, differ by about 1e-4 in UAUC). Every
statistic is computed from stored per-pair scores with seed 0; bootstraps resample users with 2,000 resamples.
