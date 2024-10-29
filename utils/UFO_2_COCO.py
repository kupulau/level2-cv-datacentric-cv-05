import json
from typing import Dict, Any
from collections import Counter
import os
import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ufo_path', default='/data/ephemeral/home/data/chinese_receipt/ufo/train.json', help='path to ufo format json file')
    parser.add_argument('--coco_path', default='/data/ephemeral/home/data/chinese_receipt/ufo/coco_train.json', help='where to save coco format json file')
    parser.add_argument('--use_seg', type=bool, default=True, help='whether to use seg attribute')
    
    args = parser.parse_args()
    return args
    
    
def convert_to_coco_format(data: Dict[str, Any], use_seg: bool):
    if use_seg:
        print("segmentation attribute를 이용해서 COCO format json을 생성합니다.")
    else:
        print("coco format 변환 시 segmentation attribute를 사용하지 않습니다")
        print("변환 시 4개의 꼭짓점으로 만들어지는 가장 큰 직사각형이 새로운 box로 결정됩니다. 주의하세요.")
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}],
    }
    
    image_id_counter = 1
    annotation_id_counter = 1

    for file_name, file_data in data["images"].items():
        image_id = image_id_counter

        coco_image = {
            "id": image_id,
            "width": file_data["img_w"],
            "height": file_data["img_h"],
            "file_name": file_name,
            "license": 123,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": "2024-10-28 11:11:11"
        }
        coco_data["images"].append(coco_image)

        for word_id, word_data in file_data["words"].items():
            annotation_id = annotation_id_counter
            [tl, tr, br, bl] = word_data["points"]
            width = max(tl[0], tr[0], br[0], bl[0]) - min(tl[0], tr[0], br[0], bl[0])
            height = max(tl[1], tr[1], br[1], bl[1]) - min(tl[1], tr[1], br[1], bl[1])
            x = min(tl[0], tr[0], br[0], bl[0])
            y = min(tl[1], tr[1], br[1], bl[1])
            
            if use_seg:
                seg = [tl+tr+br+bl]
            else:
                seg = []
            
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": seg,
                "area": width * height,
                "bbox": [x, y, width, height],
                "iscrowd": 0  
            }
            coco_data["annotations"].append(coco_annotation)

            annotation_id_counter += 1  # 새로운 word 마다 +1

        image_id_counter += 1  # 새로운 image 마다 +1

    return coco_data


def main():
    args = parse_argument()
    ufo_path = args.ufo_path
    coco_path = args.coco_path
    use_seg = args.use_seg
    
    with open(ufo_path, encoding='UTF8') as f:
        data = json.load(f)
    print(f"{ufo_path}로부터 ufo format json file을 로드하였습니다.")
        
    coco_data = convert_to_coco_format(data, use_seg)
    
    # Save COCO json
    with open(coco_path, "w", encoding='UTF8') as f:
        json.dump(coco_data, f)
    print(f"{coco_path}에 coco format json file을 저장하였습니다.\n\n")


if __name__ == "__main__":
    main()