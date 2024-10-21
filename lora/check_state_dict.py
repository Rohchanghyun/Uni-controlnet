import torch
from safetensors import safe_open
from collections import defaultdict

# def print_lora_structure(file_path):
#     with safe_open(file_path, framework="pt", device="cpu") as f:
#         layer_structure = defaultdict(list)
#         for key in f.keys():
#             parts = key.split('.')
#             layer_name = '.'.join(parts[:-1])
#             layer_structure[layer_name].append(parts[-1])

#         for layer, components in layer_structure.items():
#             print(f"계층: {layer}")
#             for component in components:
#                 print(f"  - {component}")
#             print("---")

# if __name__ == "__main__":
#     lora_file_path = "pytorch_lora_weights.safetensors"
#     print_lora_structure(lora_file_path)
#     print("LoRA 가중치 구조 출력이 완료되었습니다.")


def print_ckpt_structure(file_path):
    state_dict = torch.load(file_path, map_location="cpu")
    layer_structure = defaultdict(list)
    
    for key in state_dict.keys():
        parts = key.split('.')
        layer_name = '.'.join(parts[:-1])
        layer_structure[layer_name].append(parts[-1])

    for layer, components in layer_structure.items():
        print(f"계층: {layer}")
        for component in components:
            print(f"  - {component}")
        print("---")

if __name__ == "__main__":
    ckpt_file_path = "../ckpt/v1-5-pruned.ckpt"
    print_ckpt_structure(ckpt_file_path)
    print("체크포인트 가중치 구조 출력이 완료되었습니다.")
