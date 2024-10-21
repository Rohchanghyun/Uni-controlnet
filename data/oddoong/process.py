from PIL import Image
import numpy as np

# 이미지 열기
image_path = '/workspace/mnt/sda/changhyun/Uni-ControlNet/data/oddoong/homer.jpg'
image = Image.open(image_path)

# 이미지를 NumPy 배열로 변환
img_array = np.array(image)

# 배경을 흰색으로 채우기
if img_array.ndim == 2:  # 흑백 이미지인 경우
    img_array = np.stack((img_array,) * 3, axis=-1)  # 3채널로 변환
elif img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA 이미지인 경우
    img_array = img_array[:,:,:3]  # 알파 채널 제거

# 배경을 흰색으로 채우기
img_array[img_array == 0] = 255  # 검은색(0)을 흰색(255)으로 변경

# 흰색 배경 생성
white_background = np.ones_like(img_array) * 255

# 이미지와 흰색 배경 합성
result = np.where(img_array == 0, white_background, img_array)

# 결과를 PIL Image로 변환
result_image = Image.fromarray(result.astype('uint8'))

# 결과 저장
result_image.save('homer_white_background.jpg')

print("이미지가 성공적으로 처리되어 저장되었습니다.")
