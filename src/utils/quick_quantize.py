from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained('Qwen/Qwen3-14B', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B', trust_remote_code=True)

quant_config = {
  "zero_point": True,
  "q_group_size": 128,
  "w_bit": 4,
  "version": "GEMM"
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized('models/Quantized/Qwen3-14B-8bit/')
tokenizer.save_pretrained('models/Quantized/Qwen3-14B-8bit/')