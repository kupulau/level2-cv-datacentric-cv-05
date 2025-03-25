import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def split_ufo_dataset(data_path, train_path, val_path, train_ratio=0.8):
    # 1. 데이터 로드 및 검증
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise Exception(f"파일을 찾을 수 없습니다: {data_path}")
    except json.JSONDecodeError:
        raise Exception(f"올바른 JSON 형식이 아닙니다: {data_path}")
    
    if 'images' not in data:
        raise ValueError("Invalid UFO format: 'images' key not found")
    
    images = list(data['images'].keys())
    print(f"Total images: {len(images)}")
    
    if len(images) == 0:
        raise ValueError("데이터셋이 비어있습니다")
    
    # 2. 데이터 분할
    train, val = train_test_split(
        images, 
        train_size=train_ratio, 
        shuffle=True,
        random_state=42
    )
    
    # 3. 데이터셋 생성
    train_images = {img_id: data['images'][img_id] for img_id in train}
    val_images = {img_id: data['images'][img_id] for img_id in val}
    
    print(f"Train set: {len(train_images)} images ({train_ratio*100}%)")
    print(f"Val set: {len(val_images)} images ({(1-train_ratio)*100}%)")
    
    # 4. 저장
    try:
        for path, dataset in [(train_path, train_images), (val_path, val_images)]:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'images': dataset}, f, ensure_ascii=False, indent=4)
            print(f"저장 완료: {path}")
            
    except Exception as e:
        raise Exception(f"파일 저장 중 에러 발생: {str(e)}")
            
    return len(train_images), len(val_images)

def process_language_dataset(base_dir, language, train_ratio=0.8):
    """각 언어별 데이터셋 처리"""
    
    ### 여기에 나눌 Data json 이름이랑, train, val 파일의 이름을 정합니다.
    
    data_dir = os.path.join(base_dir, f'{language}_receipt/ufo')
    read_train_json = os.path.join(data_dir, 'train_first.json')
    write_train_json = os.path.join(data_dir, 'train2.json')
    write_val_json = os.path.join(data_dir, 'val2.json')
    
    print(f"\n=== Processing {language} dataset ===")
    try:
        train_size, val_size = split_ufo_dataset(
            data_path=read_train_json,
            train_path=write_train_json,
            val_path=write_val_json,
            train_ratio=train_ratio
        )
        
        print(f"{language} 데이터셋 분할 완료:")
        print(f"- Train: {train_size} images")
        print(f"- Val: {val_size} images")
        return True
        
    except Exception as e:
        print(f"{language} 처리 중 에러 발생: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/ephemeral/home/code/data')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()
    
    # 처리할 언어 리스트
    languages = ['chinese', 'japanese', 'thai', 'vietnamese']
    
    # 각 언어별 처리
    results = []
    for lang in languages:
        success = process_language_dataset(
            base_dir=args.data_dir,
            language=lang,
            train_ratio=args.train_ratio
        )
        results.append((lang, success))
    
    # 최종 결과 출력
    print("\n=== 처리 결과 요약 ===")
    for lang, success in results:
        status = "성공" if success else "실패"
        print(f"{lang}: {status}")

if __name__ == '__main__':
    main()