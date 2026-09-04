# ewm-nanolm

## The 5th EWM front end: a traditional LLM

Four demos already show the Emerging World Model boundary inside different
AI systems:

| Demo | Front end | Project |
| --- | --- | --- |
| 1 | nanoLM | `ewm-cortex` / `EWM-nanoLM` |
| 2 | DeepSeek-OCR | `hllset_cortex` |
| 3 | V-JEPA (video world model) | `ewm-jepa` |
| 4 | NVIDIA VLA (Cosmos-Policy) | `ewm-vla` |
| 5 | **Traditional LLM** | **`ewm-nanolm`** (this project) |

This is the **simplest and most faithful case**: a traditional causal LLM
already speaks in discrete token IDs.  No quantizer is needed — the
tokenizer's output *is* the encoding stream.

The model is **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** — a 1.5B
decoder-only LLM that runs comfortably on the RTX 3060 (12 GB) in fp16
(weights are already cached on this machine).

## The EWM↔LLM boundary

```text
prompt ──► tokenizer ──► token IDs  "tid671 tid18308 tid4854 ..."
                           │
                           ▼  HLLSet lattice (IICA)
                gate_TF = the model's vocabulary (world view)
                LUT accumulates TF for every tid the model has seen
                DRN / BSS / temporal pyramid = holographic memory of text
                           │
                           ▼  materialize → restored "tid..." stream
                           │
                           ▼  detokenize → restored prompt
                           │
                           ▼
                        LLM ──► generated text
```

Same story, fifth front end: the EWM boundary is **encoding-agnostic**
(Principle 2, IICA morphisms).  BPE token IDs need no quantizer — this is
the exact `hllset_cortex` pattern, minus the OCR encoder.

## Layout

```text
ewm-nanolm/
├── vendor/
│   └── hllset_cortex/   # vendored HLLSet stack (Python pkg + Rust crate)
├── captures/            # captured LLM token-id streams (gitignored)
├── ewm_nanolm/          # token-id pipeline: gate=vocab, LUT, BSS, DRN
├── scripts/
│   ├── capture_llm.py           # run the 1.5B LLM once, save token streams
│   └── build_notebook.py
├── notebooks/
│   └── 01_nanolm_hllset_pipeline.ipynb
└── README.md
```

## Setup

```bash
# 1. conda env (PyTorch cu124 + transformers)
conda create -n ewm-nanolm python=3.10 -y -c conda-forge
/home/alexmy/.conda/envs/ewm-nanolm/bin/pip install \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
/home/alexmy/.conda/envs/ewm-nanolm/bin/pip install transformers accelerate

# 2. HLLSet stack — vendored in this repo
cd /home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-nanolm
maturin build --release -q \
    --manifest-path vendor/hllset_cortex/crates/hllset_py/Cargo.toml
/home/alexmy/.conda/envs/ewm-nanolm/bin/pip install \
    vendor/hllset_cortex/crates/hllset_py/target/wheels/hllset_py-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
/home/alexmy/.conda/envs/ewm-nanolm/bin/pip install -e vendor/hllset_cortex

# 3. Capture real LLM token streams (RTX 3060)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    /home/alexmy/.conda/envs/ewm-nanolm/bin/python scripts/capture_llm.py

# 4. Run the notebook (kernel: Python 3 (ewm-nanolm))
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    /home/alexmy/.conda/envs/ewm-nanolm/bin/jupyter notebook notebooks/
```

## References

- **EWM consolidation:** `ewm-jepa/docs/EWM/main.tex`
- **HLLSet theory:** A. Mylnikov, *HLLSet Theory*, ASTESJ 11(2), 2026
- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (MIT license)
