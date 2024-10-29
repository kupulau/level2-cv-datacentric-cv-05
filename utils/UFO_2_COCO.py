import json
from typing import Dict, Any
from collections import Counter
import os


def parse_argument():
    pass
def convert_to_coco_format(data: Dict[str, Any]):
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
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [],
                "area": width * height,
                "bbox": [x, y, width, height],
                "iscrowd": 0  
            }
            coco_data["annotations"].append(coco_annotation)

            annotation_id_counter += 1  # 새로운 word 마다 +1

        image_id_counter += 1  # 새로운 image 마다 +1

    return coco_data


def main():
    with open("./data/thai_receipt/ufo/train.json", encoding='UTF8') as f:
        data = json.load(f)
        
    coco_data = convert_to_coco_format(data)
    
    # Save COCO json
    with open("./data/thai_receipt/ufo/train_coco.json", "w", encoding='UTF8') as f:
        json.dump(coco_data, f)


if __name__ == "__main__":
    main()