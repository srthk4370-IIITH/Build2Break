from pdf_to_video_pipeline import orchestrate_pipeline, PipelineConfig

cfg = PipelineConfig(pdf_path="tests/sample_inputs/sample.pdf")
result = orchestrate_pipeline(cfg)

assert "video_script" in result["unified_prompt"]
assert "audio_script" in result["unified_prompt"]
print("Test passed!")