import sys
if './' not in sys.path:
	sys.path.append('./')
from utils.share import *
import utils.config as config

import cv2
import einops
import imghdr
import pdb
import numpy as np
import os

import torch
from pytorch_lightning import seed_everything
import torchvision.transforms as transforms

from PIL import Image, ImageFilter
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
from annotator.uniformer import UniformerDetector
from annotator.content import ContentDetector
from safetensors import safe_open
from huggingface_hub import hf_hub_download

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler

CUDA_VISIBLE_DEVICES=1
def print_model_modules(model):
    print("Uni-controlnet의 모듈:")
    for name, module in model.named_modules():
        print(f"- {name}")
        if isinstance(module, torch.nn.Sequential) and len(list(module.children())) > 0:
            last_layer = list(module.children())[-1]
            if hasattr(last_layer, 'in_channels') and hasattr(last_layer, 'out_channels'):
                print(f"  마지막 레이어 ({name}) 입력 채널: {last_layer.in_channels}")
                print(f"  마지막 레이어 ({name}) 출력 채널: {last_layer.out_channels}")
        
        # 다운샘플링 레이어 확인
        if isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)) or \
           (isinstance(module, torch.nn.Conv2d) and module.stride[0] > 1):
            print(f"  다운샘플링 레이어: {name} ({type(module).__name__})")

def apply_lora_weights(model, lora_checkpoint_path, lora_scale=1.0, custom_mappings=None):
    print(f"LoRA 가중치를 {lora_checkpoint_path}에서 로딩 중")
    
    with safe_open(lora_checkpoint_path, framework="pt", device="cuda") as f:
        lora_weights = {key: f.get_tensor(key) for key in f.keys()}

    model_dict = model.state_dict()
    # 적용 전 일부 파라미터 출력
    sample_key = list(model_dict.keys())[0]
    print(f"LoRA 적용 전: {sample_key} = {model_dict[sample_key].mean().item()}")

    # LoRA 가중치를 순회하며 모델의 state dict 업데이트
    def map_lora_to_model_key(lora_key):
        if custom_mappings:
            for lora_prefix, model_prefix in custom_mappings.items():
                if lora_key.startswith(lora_prefix):
                    suffix = lora_key[len(lora_prefix):]
                    parts = suffix.split('.')
                    if len(parts) >= 3 and parts[0] == "transformer_blocks":
                        block_num = int(parts[1])
                        component = '.'.join(parts[2:])
                        return f"{model_prefix}transformer_blocks.{block_num}.{component}"
        return lora_key

    for lora_key in lora_weights:
        if '.lora.up.' in lora_key:
            base_key = lora_key.split('.lora.up.')[0]
            model_key = map_lora_to_model_key(base_key)
            
            # .weight를 유지한 키로 모델 딕셔너리에서 검색
            model_key_with_weight = model_key + '.weight'
            lora_key_with_weight = lora_key
            if model_key_with_weight in model_dict:
                up_weight = lora_weights[lora_key_with_weight]
                down_weight = lora_weights[f"{base_key}.lora.down.weight"]
                
                # LoRA 가중치 계산: up_weight * down_weight
                lora_weight = torch.matmul(up_weight, down_weight)
                
                if model_dict[model_key_with_weight].shape != lora_weight.shape:
                    print(f"형태 불일치: {lora_key_with_weight} (LoRA) vs {model_key_with_weight} (모델). 건너뜁니다.")
                    continue
                
                # LoRA 가중치에 스케일 적용하고 모델 가중치에 더하기
                model_dict[model_key_with_weight] = model_dict[model_key_with_weight] + lora_scale * lora_weight
                print(f"LoRA 적용됨: {lora_key_with_weight} (LoRA) -> {model_key_with_weight} (모델)")
            else:
                print(f"키를 찾을 수 없음: {lora_key_with_weight} (LoRA) -> {model_key_with_weight} (모델). 건너뜁니다.")

    # 업데이트된 state dict 로드
    model.load_state_dict(model_dict)

    # 적용 후 같은 파라미터 출력
    print(f"LoRA 적용 후: {sample_key} = {model_dict[sample_key].mean().item()}")

    print(f"LoRA 가중치가 {lora_checkpoint_path}에서 로드되어 모델에 적용되었습니다.")
    return model

# 사용 예시:
custom_mappings = {
    "unet.down_blocks.0.attentions.0.": "model.diffusion_model.input_blocks.1.1.",
    "unet.down_blocks.0.attentions.1.": "model.diffusion_model.input_blocks.2.1.",
    "unet.down_blocks.1.attentions.0.": "model.diffusion_model.input_blocks.4.1.",
    "unet.down_blocks.1.attentions.1.": "model.diffusion_model.input_blocks.5.1.",
    "unet.down_blocks.2.attentions.0.": "model.diffusion_model.input_blocks.7.1.",
    "unet.down_blocks.2.attentions.1.": "model.diffusion_model.input_blocks.8.1.",
    
    "unet.mid_block.attentions.0.": "model.diffusion_model.middle_block.1.",
    
    "unet.up_blocks.1.attentions.0.": "model.diffusion_model.output_blocks.3.1.",
    "unet.up_blocks.1.attentions.1.": "model.diffusion_model.output_blocks.4.1.",
    "unet.up_blocks.1.attentions.2.": "model.diffusion_model.output_blocks.5.1.",
    "unet.up_blocks.2.attentions.0.": "model.diffusion_model.output_blocks.6.1.",
    "unet.up_blocks.2.attentions.1.": "model.diffusion_model.output_blocks.7.1.",
    "unet.up_blocks.2.attentions.2.": "model.diffusion_model.output_blocks.8.1.",
    "unet.up_blocks.3.attentions.0.": "model.diffusion_model.output_blocks.9.1.",
    "unet.up_blocks.3.attentions.1.": "model.diffusion_model.output_blocks.10.1.",
    "unet.up_blocks.3.attentions.2.": "model.diffusion_model.output_blocks.11.1.",
    
    # lora_dict : model_dict
}


def load_lora_weights(model, repo_id, filename="pytorch_lora_weights.safetensors"):
    """
    Load LoRA weights from Hugging Face model repository.

    Args:
        model: The base model to apply the LoRA to.
        repo_id: The Hugging Face model repository ID.
        filename: The filename of the LoRA weights file.
    """
    print(f"Downloading LoRA weights from {repo_id}")
    
    # Download the file
    cached_file = hf_hub_download(repo_id=repo_id, filename=filename)
    
    print(f"LoRA weights downloaded to {cached_file}")
    
    # Load the weights
    with safe_open(cached_file, framework="pt", device="cuda") as f:
        lora_weights = {key: f.get_tensor(key) for key in f.keys()}

    model_dict = model.state_dict()
    
    # Iterate through the LoRA weights and update the model's state dict
    for key in lora_weights:
        if key in model_dict:
            if model_dict[key].shape != lora_weights[key].shape:
                print(f"Shape mismatch for key {key}. Skipping.")
                continue
            model_dict[key] = model_dict[key] + lora_weights[key]
        else:
            print(f"Key {key} not found in model. Skipping.")

    # Load the updated state dict
    model.load_state_dict(model_dict)

    print(f"LoRA weights from {repo_id} applied to the model.")
    
apply_canny = CannyDetector()
apply_mlsd = MLSDdetector()
apply_hed = HEDdetector()
apply_sketch = SketchDetector()
apply_openpose = OpenposeDetector()
apply_midas = MidasDetector()
apply_seg = UniformerDetector()
apply_content = ContentDetector()

lora_checkpoint_path = "/workspace/mnt/sda/changhyun/dreambooth_lora/lora/oddoong/total_no_text/t2i/data_preprocess_identifier_1337_cosine_no_crop_4000/checkpoint-2000/pytorch_lora_weights.safetensors"
repo_id = "lora-library/B-LoRA-cartoon_line"

model = create_model('./configs/uni_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./ckpt/uni.ckpt', location='cuda'))

model = model.cuda()

# add LoRA
#model = apply_lora_weights(model, lora_checkpoint_path, lora_scale=1.0, custom_mappings=custom_mappings)
#load_lora_weights(model, repo_id)

ddim_sampler = DDIMSampler(model)

# Print model modules
#print_model_modules(model.model)
# # 모델 구조를 문자열로 캡처
# model_architecture = str(model.model)

# # architecture.txt 파일에 모델 구조 저장
# with open('architecture.txt', 'w') as f:
#     f.write(model_architecture)

# print("모델 구조가 architecture.txt 파일에 저장되었습니다.")

def save_detected_maps(detected_maps_list, save_dir, sample_idx):
    map_names = ["canny", "mlsd", "hed", "sketch", "openpose", "midas", "seg"]
    
    # 저장할 디렉토리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # detected_maps_list가 비어 있을 경우 처리
    if len(detected_maps_list) == 0:
        print(f"No detected maps for sample {sample_idx}. Skipping...")
        return
    
    # 각각의 맵을 순차적으로 저장
    for i, detected_map in enumerate(detected_maps_list):
        if i >= len(map_names):
            print(f"Skipping detected map index {i} as it exceeds map_names list.")
            continue  # 예외처리: map_names 리스트보다 detected_maps_list가 길 때
        
        # (H, W, C) 배열을 (W, H, C)로 변환하여 PIL 이미지로 변환
        if detected_map is not None and detected_map.size > 0:  # 이미지가 있을 경우에만 저장
            img_pil = Image.fromarray(np.uint8(detected_map))
            
            # 샘플별로 파일 경로 생성 및 저장
            img_pil.save(os.path.join(save_dir, f'{map_names[i]}_detected_sample_{sample_idx}.png'))
            print(f"Saved {map_names[i]} image for sample {sample_idx} to: {os.path.join(save_dir, f'{map_names[i]}_detected_sample_{sample_idx}.png')}")
        else:
            print(f"Detected map {map_names[i]} for sample {sample_idx} is empty. Skipping...")


def process(canny_image, mlsd_image, hed_image, sketch_image, openpose_image, midas_image, seg_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength):
    
    seed_everything(seed)
    # condition image 받아와 HWC(3) 형태로 reshape 후 저장
    if canny_image is not None:
        anchor_image = canny_image
    elif mlsd_image is not None:
        anchor_image = mlsd_image
    elif hed_image is not None:
        anchor_image = hed_image
    elif sketch_image is not None:
        anchor_image = sketch_image
    elif openpose_image is not None:
        anchor_image = openpose_image
    elif midas_image is not None:
        anchor_image = midas_image
    elif seg_image is not None:
        anchor_image = seg_image
    elif content_image is not None:
        anchor_image = content_image
    else:
        anchor_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape
    
    # input으로 들어온 condition image들을 condition에 맞게 변환하여 저장
    with torch.no_grad():
        if canny_image is not None:
            canny_image = cv2.resize(canny_image, (W, H))
            canny_detected_map = HWC3(apply_canny(HWC3(canny_image), low_threshold, high_threshold))
        else:
            canny_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if mlsd_image is not None:
            mlsd_image = cv2.resize(mlsd_image, (W, H))
            mlsd_detected_map = HWC3(apply_mlsd(HWC3(mlsd_image), value_threshold, distance_threshold))
        else:
            mlsd_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if hed_image is not None:
            hed_image = cv2.resize(hed_image, (W, H))
            hed_detected_map = HWC3(apply_hed(HWC3(hed_image)))
        else:
            hed_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if sketch_image is not None:
            sketch_image = cv2.resize(sketch_image, (W, H))
            sketch_detected_map = HWC3(apply_sketch(HWC3(sketch_image)))
            sketch_detected_map = Image.fromarray(sketch_detected_map)
            sketch_detected_map = sketch_detected_map.filter(ImageFilter.GaussianBlur(radius=5)) 
            sketch_detected_map.save(os.path.join("/workspace/projects/emoji_generation/Uni-controlnet/data/oddoong/lora/blur_10_sketch_content_identifier_action_paduk", 'blurred_sketch.png'))
            sketch_detected_map = np.array(sketch_detected_map)           
        else:
            sketch_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if openpose_image is not None:
            openpose_image = cv2.resize(openpose_image, (W, H))
            openpose_detected_map, _ = apply_openpose(HWC3(openpose_image), False)
            openpose_detected_map = HWC3(openpose_detected_map)
        else:
            openpose_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if midas_image is not None:
            midas_image = cv2.resize(midas_image, (W, H))
            midas_detected_map = HWC3(apply_midas(HWC3(midas_image), alpha))
        else:
            midas_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if seg_image is not None:
            seg_image = cv2.resize(seg_image, (W, H))
            seg_detected_map, _ = apply_seg(HWC3(seg_image))
            seg_detected_map = HWC3(seg_detected_map)
        else:
            seg_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if content_image is not None:
            content_emb = apply_content(content_image)
        else:
            content_emb = np.zeros((768))

        detected_maps_list = [canny_detected_map, 
                              mlsd_detected_map, 
                              hed_detected_map,
                              sketch_detected_map,
                              openpose_detected_map,
                              midas_detected_map,
                              seg_detected_map                          
                              ]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        # input으로 받은 condition들을 concat 
        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]} # global, local condition # 이부분을 PAG로 변경해야함
        shape = (4, H // 8, W // 8)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, global_strength=global_strength) # global, local condition 추가

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]

    return [results, detected_maps_list]


canny_image = None #np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/condition.png").convert('RGB'))
mlsd_image = None #np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/condition.png").convert('RGB'))
hed_image = None #np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/condition.png").convert('RGB'))
sketch_image = None #np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/sketch.png").convert('RGB'))
openpose_image = None #np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/condition.png").convert('RGB'))
midas_image = None #np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/condition.png").convert('RGB'))
seg_image = np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/condition.png").convert('RGB'))
content_image = np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/sketch.png").convert('RGB'))
#np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/sketch.png").convert('RGB'))
#np.array(Image.open("/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/condition.png").convert('RGB'))
prompt = "a white duck with yellow beak laughing"
a_prompt = "best quality, extremely detailed"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
num_samples = 5
image_resolution = 512
ddim_steps = 50
strength = 1
scale = 7.5
seed = 733
eta = 0.0
low_threshold = 100
high_threshold = 200
value_threshold = 0.1
distance_threshold = 0.1
alpha = 6.2
global_strength = 1

save_dir = "/workspace/projects/emoji_generation/Uni-controlnet/samples/condition_test/seg"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


ips = [canny_image, mlsd_image, hed_image, sketch_image, openpose_image, midas_image, seg_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength]




outputs, maps = process(canny_image, mlsd_image, hed_image, sketch_image, openpose_image, midas_image, seg_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength )


temp_sketch = Image.fromarray(np.array(maps[3]))  # numpy.ndarray를 PIL 이미지로 변환
temp_sketch.save(os.path.join(save_dir, 'blurred_sketch.png'))

# 주어진 리스트를 NumPy 배열로 변환
image_array = np.array(outputs)

# 이미지를 순차적으로 저장
for i in range(5):
    # 각각의 이미지를 (3, 512, 640)로 변환
    img = image_array[i].transpose(2, 1, 0)  # (640, 512, 3) -> (3, 512, 640)
    
    # 이미지를 (512, 640, 3)으로 변환하여 저장 (PIL은 채널이 마지막에 와야 함)
    img_pil = Image.fromarray(np.uint8(img.transpose(2, 1, 0)))  # 다시 (512, 640, 3)로 변환
    
    # 파일 경로 생성 및 저장 (이미지 번호로 이름 생성)
    img_pil.save(os.path.join(save_dir, f'image_canny_{i+6}.png'))