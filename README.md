# Build2Break
PDF to 1-2 min animated video pipeline

Architecture: pipeline stages (PDF → Summarize → Prompt → TTS → JSON2Video → FFmpeg) ✅

Models used: HuggingFace BART for summary, Flan-T5 for prompt, ElevenLabs for TTS ✅

Datasets: User PDF input (any PDF) ✅

Safety measures: JSON validation, sanitization, asset whitelisting ✅

Known limitations: Needs API keys, JSON2Video demo is mocked, TTS upload required ✅
