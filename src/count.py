import json

def count_unique_images(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 집합(set)을 생성하여 중복을 자동으로 제거
    unique_images = set()
    
    for item in data:
        if "reference_image" in item:
            unique_images.add(item["reference_image"])
        if "target_image" in item:
            unique_images.add(item["target_image"])
            
    return len(unique_images)

# 파일 경로를 입력하세요
file_path = '/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/meta_data/macir_meta_full.json'
total_count = count_unique_images(file_path)
print(f"전체 고유 이미지 개수: {total_count}")