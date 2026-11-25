# Jietian: MobileNetV3-Small and efficiency benchmark

This folder contains my training code, the validated MobileNetV3-Small results for all three training fractions, and the shared architecture efficiency benchmark.

## MobileNetV3-Small results

| Training fraction | Best epoch | Validation macro AUC | Test macro AUC | Training time (s) |
| --- | ---: | ---: | ---: | ---: |
| 100% | 8 | 0.707787 | 0.670781 | 390.2 |
| 50% | 5 | 0.665776 | 0.632360 | 245.5 |
| 25% | 7 | 0.622252 | 0.599345 | 175.3 |

Download and extract [mobilenet_submission.zip](mobilenet_submission.zip). Each fraction includes validation/test prediction probabilities, a pure `state_dict` best checkpoint, epoch logs, training information, configuration, history and run summary. `mobilenet_validation_report.json` records the independent checks. The full 30 epoch checkpoints remain archived locally; the compact submission retains the selected best weights for each run.

Training used Nathan's shared patched `train.py`, ImageNet pretrained weights, 14 labels, RGB 224 x 224, batch size 32, 10 epochs, seed 2026, AdamW lr 1e-4 and weight decay 1e-4. These are the later patched-script settings, which differ from the earlier suggested seed 42 and weight decay 0.01. Validation/test were fixed across fractions. Training image counts were 7,263 / 3,632 / 1,828. Images used a lossless RGB/resize cache; all pixel hashes and 16 original-image torchvision comparisons were verified before training. Training flip and normalization stayed in the shared trainer.

`train.py` is the unmodified shared trainer. `run_mobilenet.py` checks the data and exports standardized filenames. The training notebook and `requirements.txt` are included. Use the team-shared five CSVs and NIH images; machine-specific paths must be adjusted.

## Architecture efficiency benchmark

All three architectures were measured in the same Google Colab runtime: Intel Xeon CPU @ 2.00 GHz (2 allocated logical CPUs), Tesla T4, PyTorch 2.14.1+cu130, timm 1.0.31. Four PyTorch CPU threads follow Nathan's supplied benchmark.

| Architecture | Parameters (M) | FLOPs (G) | GMACs | CPU latency (ms) | T4 latency (ms) | Inference peak allocated GPU memory (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 11.184 | 3.627 | 1.814 | 62.81 | 2.48 | 77.89 |
| MobileNetV3-Small | 1.532 | 0.111 | 0.056 | 26.40 | 5.13 | 18.07 |
| DeiT-Tiny | 5.527 | 2.507 | 1.253 | 55.50 | 7.06 | 31.54 |

Use [efficiency_benchmark.csv](efficiency_benchmark.csv) for full-precision numbers, including timing standard deviations. `forward_flops` is the raw per-forward count, `gflops = forward_flops / 1e9`, and Nathan's `gmacs = forward_flops / 2e9`. FLOPs (G) is operation count, not achieved FLOP/s.

Protocol: eager FP32 eval with `torch.no_grad`, batch 1, input 1 x 3 x 224 x 224, 14 outputs, 10 warmups, five trials of 50 timed forwards. CUDA is synchronized around each timed block. Each architecture uses a fresh subprocess. Latency excludes image loading, preprocessing and host/device transfer. No checkpoint or patient image is needed for architecture-level inference benchmarking; random initialization follows Nathan's code. This does not evaluate accuracy.

DeiT FLOP profiling explicitly counts attention matrix multiplications because the installed counter omitted the fused CPU attention path. Its saved latency and memory use the original default fused path. The counter estimates registered operators; it is not an exhaustive count of every elementary operation. Memory is PyTorch peak allocated tensor memory, includes model/input/inference allocations, and excludes CUDA context and non-PyTorch memory. These cloud measurements are not Raspberry Pi measurements.

Reproduce with the standalone `benchmark_efficiency.py` or `Efficiency_Benchmark_Colab.ipynb` on a CUDA runtime. Per-model JSON files save hardware information, individual trial means and counted operators. See `benchmark_protocol.json` and `validation.json`.

## Shared training-time collection

`training_runs.csv` / `.json` contain seven received records: ResNet-18 100/50/25, MobileNetV3-Small 100/50/25 and DeiT-Tiny 100. Nathan's DeiT 50/25 summaries had not been received at collection time and are marked missing, not zero. Training peak GPU memory is separate from inference peak memory. Training times were collected from separate training sessions and are descriptive rather than a controlled common-machine training-speed comparison.
