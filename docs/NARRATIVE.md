# EWM × LLM — The 5th Front End

Canonical narrative: [`ewm-jepa/docs/EWM/main.tex`](../ewm-jepa/docs/EWM/main.tex)
and [`ewm-jepa/docs/NARRATIVE.md`](../ewm-jepa/docs/NARRATIVE.md).  This file
is the translation layer for a **traditional causal LLM**.

## The thesis

A traditional LLM is the case EWM was first built for: the tokenizer already
emits discrete IDs, so the boundary needs **no quantizer**.  The LLM
contributes a world-class language model; EWM contributes the emergent
ontology (the LUT over token IDs), the content-addressed gate (the model's
vocabulary as world view), the holographic memory of text, and the
gradient-free executive.  **The LLM predicts the next token; EWM remembers
everything the LLM has ever encoded.**

## Mapping the four principles

| EWM principle | LLM instantiation |
| --- | --- |
| 1. Emergent ontology | An element is a token ID that persists across prompts in the LUT — not the fixed 151k vocab, but what the model *actually uses*. |
| 2. IICA morphisms | token IDs → MurmurHash3 → HLLSet bits → gate ∩ → LUT → materialize; the round-trip is IICA between commits. |
| 3. Ashby–Bootstrap | n-gram tokenization (1-, 2-, 3-grams over the token-id stream) multiplies measurements — the same Khayyam bootstrap as OCR and JEPA. |
| 4. Noether evolution | DRN over prompt streams: Novel = new vocabulary entering the context, Departed = fading topics, Retained = the model's active world view. The union invariant = no prompt is ever forgotten. |

## The perceptron taxonomy, LLM edition

| Tier | ds-hllset-cortex | ewm-jepa | ewm-vla | **ewm-nanolm** |
| --- | --- | --- | --- | --- |
| Type 0 — substrate | OCR encoder + MurmurHash3 | V-JEPA encoder + quantizer | VLA + action quantizer | **tokenizer + MurmurHash3** (no quantizer) |
| Type 1 — forward | tokenizer + gate + LUT | tokenizer + gate + LUT | n-gram + safety gate + action LUT | **n-gram + vocab gate + token LUT** |
| Type 2 — executive | materializer + OCR decoder | materializer + predictor | materializer + unquantizer | **materializer + LLM (generation)** |

## Why this is the anchor demo

Every other front end had to be *made* discrete (quantized).  The LLM is
natively discrete — which is the strongest evidence for the central claim:
**the EWM boundary is encoding-agnostic because it was designed around the
discrete case first.**  ds-hllset-cortex is this demo with a vision front
end; ewm-jepa and ewm-vla are this demo with continuous front ends that we
re-discretized.  ewm-nanolm closes the loop.

## The data boundary

```text
prompt ──► tokenizer ──► token IDs "tid671 tid18308 ..."
                            │
                            ▼  HLLSet lattice (IICA)
                 gate_TF = model vocabulary ∩ · LUT · DRN · BSS · pyramid
                            │
                            ▼  materialize (TF-ranked, De Bruijn)
                            │
                            ▼  restored token IDs
                            │
                            ▼
                   LLM generation ──► text
```

Only the front end changed.  Again.
