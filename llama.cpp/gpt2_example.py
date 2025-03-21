# from transformers import AutoTokenizer, AutoModelForCausalLM

# HF_REPO_NAME = "deepseek-ai/deepseek-llm-7b-base"

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

# tokenizer.save_pretrained('deepseek-llm')
# model.save_pretrained('deepseek-llm')

from transformers import AutoTokenizer, AutoModelForCausalLM

HF_REPO_NAME = "yuhuili/EAGLE-Vicuna-7B-v1.3"

# 모델 로드
model_path = "yuhuili/EAGLE-Vicuna-7B-v1.3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 모델 저장
save_dir = r"/home/youngmin/llama.cpp/EAGLE"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
