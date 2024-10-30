import os
import json
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

def read_json(filename: str):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def save_vis_to_img(save_dir: str | os.PathLike, inference_dir: str | os.PathLike = 'output.csv', image_dir: str | os.PathLike = 'data') -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)    
    data = read_json(inference_dir)
    
    for im, points in data['images'].items():
        # 이미지 경로를 직접 지정된 폴더에서 찾음
        im_path = Path(image_dir) / im
        
        if not im_path.exists():
            print(f"Warning: Image {im} not found in {image_dir}")
            continue
            
        img = Image.open(im_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        draw = ImageDraw.Draw(img)
        
        for obj_k, obj_v in points['words'].items():
            # bbox points
            pts = [(int(p[0]), int(p[1])) for p in obj_v['points']]
            pt1 = sorted(pts, key=lambda x: (x[1], x[0]))[0]

            draw.polygon(pts, outline=(255, 0, 0))                
            draw.text(
                (pt1[0]-3, pt1[1]-12),
                obj_k,
                fill=(0, 0, 0)
            )
        img.save(os.path.join(save_dir, im))
        
if __name__ == "__main__":
    save_vis_to_img(
        save_dir = "/data/ephemeral/home/vis_test_baseline4",
        inference_dir = "/data/ephemeral/home/vis_test_custom/output.csv",
        image_dir = "/data/ephemeral/home/Custom_thai"  # 원본 이미지가 있는 폴더 경로
    )