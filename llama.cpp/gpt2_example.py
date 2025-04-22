# from transformers import AutoTokenizer, AutoModelForCausalLM

# HF_REPO_NAME = "deepseek-ai/deepseek-llm-7b-base"

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

# tokenizer.save_pretrained('deepseek-llm')
# model.save_pretrained('deepseek-llm')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch # torch import 추가

# HF_REPO_NAME = "yuhuili/EAGLE-Vicuna-7B-v1.3"
HF_REPO_NAME = "lmsys/vicuna-7b-v1.3"

# 모델 로드
# model_path = "yuhuili/EAGLE-Vicuna-7B-v1.3"
model_path = "lmsys/vicuna-7b-v1.3"
# print(f"'{model_path}' 에서 토크나이저 로딩 중...")
# tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"'{model_path}' 에서 모델 로딩 중...")
# device_map='auto'를 추가하여 가능한 경우 GPU를 사용하도록 할 수 있습니다.
# 메모리가 부족하면 'cpu'로 명시하거나 이 옵션을 제거하세요.
# low_cpu_mem_usage=True는 CPU 메모리 사용량을 줄이는 데 도움이 될 수 있습니다.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # torch_dtype=torch.float16, # 필요시 bfloat16 또는 float16 사용 (메모리 절약)
    device_map='auto',        # 가능하면 GPU 사용
    low_cpu_mem_usage=True    # CPU 메모리 사용량 최적화 시도
)
print("모델 로딩 완료.")

# ... 모델 로드 후 ...

print("\n" + "="*50)
print("모델 웨이트 구조 (List Comprehension 사용):")
print("="*50)

state_dict = model.state_dict()
# List comprehension을 사용하여 각 파라미터 정보 포맷팅
# 내부적으로는 state_dict.items()를 순회합니다.
param_info_list = [
    f"Name: {name:<70} | Shape: {str(param.shape):<25} | Dtype: {param.dtype}"
    for name, param in state_dict.items()
]
# 리스트의 각 항목을 줄바꿈으로 연결하여 출력
print("\n".join(param_info_list))

# 총 파라미터 수도 비슷한 방식으로 계산 가능
total_params = sum(p.numel() for p in state_dict.values())
print("="*50)
print(f"총 파라미터 수: {total_params} ({total_params / 1_000_000:.2f}M)")
print("="*50 + "\n")

# # ... 모델 저장 ...

# # --- 추가된 코드: 모델 웨이트 구조 확인 ---
# print("\n" + "="*50)
# print("모델 웨이트 구조 (파라미터 이름, 형태, 데이터 타입):")
# print("="*50)
# total_params = 0
# for name, param in model.named_parameters():
#     # param.requires_grad는 해당 파라미터가 학습 가능한지 여부를 나타냅니다.
#     param_count = param.numel() # 파라미터의 총 개수
#     total_params += param_count
#     print(f"Name: {name:<70} | Shape: {str(param.shape):<25} | Dtype: {param.dtype:<15} | Requires Grad: {param.requires_grad} | Count: {param_count}")

# print("="*50)
# print(f"총 파라미터 수: {total_params}")
# # 백만 단위로 변환하여 출력
# print(f"총 파라미터 수 (백만 단위): {total_params / 1_000_000:.2f}M")
# print("="*50 + "\n")

# # (선택 사항) 모델의 전체 레이어 구조를 간단히 보려면:
# # print("모델 전체 레이어 구조:")
# # print(model)
# # print("="*50 + "\n")
# # --- 추가된 코드 끝 ---

# # 모델 저장
# save_dir = r"/home/youngmin/llama.cpp/EAGLE"
# print(f"모델과 토크나이저를 '{save_dir}' 에 저장 중...")
# # tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)
# print("저장 완료.")
