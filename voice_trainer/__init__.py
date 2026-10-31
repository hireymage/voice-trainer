"""voice-trainer — dataset + Piper TTS training tool.

TEXT → NAHRÁVKA (Mac klient) → STT KONTROLA (whisper) → DATASET (SQLite)
→ PIPER TRAINING (DDP 2× GPU) → CHECKPOINT → ONNX → PIPER INFERENCE
"""

__version__ = "0.1.0"