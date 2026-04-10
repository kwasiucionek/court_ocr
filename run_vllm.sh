export hf_model_path=dots.ocr/weights/DotsOCR-1.5
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 vllm serve $hf_model_path \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --chat-template-content-format string \
  --served-model-name model \
  --trust-remote-code
