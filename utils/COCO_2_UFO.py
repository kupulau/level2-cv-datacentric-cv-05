import json
from typing import Dict, Any
from collections import Counter


def convert_to_your_format(data: Dict[str, Any]):
    your_format = {"images": {}}

    # imd id : 파일명 형태의 dictionary
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        image_name = image_id_to_filename[image_id]
        bbox = annotation["bbox"]

        tl = [bbox[0], bbox[1]]
        tr = [bbox[0] + bbox[2], bbox[1]]
        br = [bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bl = [bbox[0], bbox[1] + bbox[3]]
        
        # COCO에서 UFO로 변환시 비는 정보는 placeholder로 대체
        if image_name not in your_format["images"]:
            your_format["images"][image_name] = {
                "paragraphs": {},
                "words": {},
                "chars": {},
                "img_w": data["images"][image_id - 1]["width"],  # img_id가 1로 시작한다고 가정
                "img_h": data["images"][image_id - 1]["height"],  # img_id가 1로 시작한다고 가정
                "tags": ["autoannotated"], 
                "relations": {},
                "annotation_log": {
                    "worker": "worker",
                    "timestamp": "2023-03-22",
                    "tool_version": "",
                    "source": None
                },
                "license_tag": {
                    "usability": True,
                    "public": False,
                    "commercial": True,
                    "type": None,
                    "holder": "Upstage"
                }
            }

        your_format["images"][image_name]["words"][str(annotation["id"]).zfill(4)] = {
            "transcription": "",  
            "points": [tl, tr, br, bl],
            "orientation": "Horizontal",  # horizontal로 가정 되고 tag가 붙여짐
            "language": None,  
            "tags": ["Auto"],  
            "confidence": None,  
            "illegibility": False 
        }

    return your_format


def main():
    # Load COCO JSON
    with open("/data/ephemeral/home/code/data/thai_receipt/ufo/thai_first_relabel_COCO.json") as f:
        coco_data = json.load(f)
    
    # UFO로 변환
    your_format_data = convert_to_your_format(coco_data)
    
    with open("/data/ephemeral/home/code/data/thai_receipt/ufo/thai_first_relabel_UFO.json", "w") as f:
        json.dump(your_format_data, f)

if __name__ == "__main__":
    main()