import json
from typing import Dict, Any
from collections import Counter
import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ufo_path', default='/data/ephemeral/home/data/chinese_receipt/ufo/new_train.json', help='where to save format json file')
    parser.add_argument('--coco_path', default='/data/ephemeral/home/data/chinese_receipt/ufo/coco_train.json', help='path to coco format json file')
    parser.add_argument('--use_seg', type=bool, default=True, help='whether to use seg attribute')
    
    args = parser.parse_args()
    return args


def convert_to_your_format(data: Dict[str, Any], use_seg: bool):
    if use_seg:
        print("segmentation attribute를 이용해서 UFO format json을 복원합니다.")
    else:
        print("COCO format bbox attribute를 이용해서 UFO format json을 복원합니다")
        print("모든 사각형이 직사각형으로 변환됨에 주의하세요!")
    
    your_format = {"images": {}}

    # imd id : 파일명 형태의 dictionary
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        image_name = image_id_to_filename[image_id]
        bbox = annotation["bbox"]
        seg = annotation['segmentation']
        id = annotation['id']

        if len(seg) and use_seg:
            tl = seg[0][:2]
            tr = seg[0][2:4]
            br = seg[0][4:6]
            bl = seg[0][6:]
        else:
            tl = [bbox[0], bbox[1]]
            tr = [bbox[0] + bbox[2], bbox[1]]
            br = [bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bl = [bbox[0], bbox[1] + bbox[3]]
            
            if use_seg:
                print(f"image {image_name}에 대해 segmentation 정보가 존재하지 않아 직사각형을 활용해 복원된 bbox가 존재합니다.")
                print(f"annotation id : {id}\n\n")
        
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
    args = parse_argument()
    ufo_path = args.ufo_path
    coco_path = args.coco_path
    use_seg = args.use_seg
    
    # Load COCO JSON
    with open(coco_path) as f:
        coco_data = json.load(f)
    print(f"{coco_path}로부터 coco format json file을 로드하였습니다.")
    
    # UFO로 변환
    your_format_data = convert_to_your_format(coco_data, use_seg)
    
    with open(ufo_path, "w") as f:
        json.dump(your_format_data, f)
        
    print(f"{ufo_path}에 ufo format json file을 저장하였습니다.\n\n")
    
    
if __name__ == "__main__":
    main()