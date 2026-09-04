# ewm_nanolm
"""EWM × traditional LLM: the HLLSet lattice around token-ID streams.

The 5th EWM front end — and the anchor case.  A causal LLM already speaks
in discrete token IDs, so no quantizer is needed: the tokenizer's output
*is* the encoding stream.  This package runs the token-ID stream through the
same lattice as OCR, JEPA, and VLA.
"""

from .pipeline import NanolmPipeline, RoundTripStats, RoundTripResult, load_capture

__all__ = ["NanolmPipeline", "RoundTripStats", "RoundTripResult", "load_capture"]
