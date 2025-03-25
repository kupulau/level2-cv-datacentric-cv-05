import os.path as osp
import sys
sys.path.append('..')
import json
import pickle
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from dataset import filter_vertices, resize_img, adjust_height, rotate_img, crop_img, generate_roi_mask
from east_dataset import generate_score_geo_maps  # EAST 관련 함수 import

def preprocessing(
    root_dir,
    split="train",
    num=0,
    image_size=2048, 
    crop_size=1024,
    ignore_under_threshold=10,
    drop_under_threshold=1,
    map_scale=0.5,  # EAST map scale 추가
):
    if crop_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']

    if num == 0:
        pkl_dir = osp.join(root_dir, "pickles/{}.pickle".format(split))  # 파일명 변경
    else:
        pkl_dir = osp.join(root_dir, "pickles/{}.pickle".format(split + str(num)))

    total_anno = dict(images=dict())
    
    # json 파일을 읽는 부분
    if num == 0:
        json_name = f'{split}.json'
    else:
        json_name = f'{split}{num}.json'

    for nation in lang_list:
        json_path = osp.join(root_dir, f'{nation}_receipt/ufo/{json_name}')
        with open(json_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)
        for im in anno['images']:
            total_anno['images'][im] = anno['images'][im]

    image_fnames = sorted(total_anno['images'].keys())

    total = dict(
        images=[],
        vertices=[],
        labels=[],
        word_bboxes=[],
        roi_masks=[],
        score_maps=[],  # EAST score maps 추가
        geo_maps=[]     # EAST geo maps 추가
    )

    for idx in tqdm(range(len(image_fnames))):
        image_fname = image_fnames[idx]
        
        lang_indicator = image_fname.split('.')[1]
        lang_map = {
            'zh': 'chinese',
            'ja': 'japanese',
            'th': 'thai',
            'vi': 'vietnamese'
        }
        lang = lang_map.get(lang_indicator)
        if not lang:
            continue
            
        if split == "val":
            split_dir = "train"
        else:
            split_dir = split
            
        image_fpath = osp.join(root_dir, f'{lang}_receipt/img/{split_dir}', image_fname)

        vertices, labels = [], []
        for word_info in total_anno['images'][image_fname]['words'].values():
            num_pts = np.array(word_info['points']).shape[0]
            if num_pts > 4:
                continue
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(1)
            
        vertices = np.array(vertices, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=ignore_under_threshold,
            drop_under=drop_under_threshold
        )

        try:
            # 이미지 전처리
            image = Image.open(image_fpath)
            image, vertices = resize_img(image, vertices, image_size)
            image, vertices = adjust_height(image, vertices)
            image, vertices = rotate_img(image, vertices)
            image, vertices = crop_img(image, vertices, labels, crop_size)

            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)

            # 기존 전처리
            word_bboxes = np.reshape(vertices, (-1, 4, 2))
            roi_mask = generate_roi_mask(image, vertices, labels)

            # EAST 데이터 생성
            score_map, geo_map = generate_score_geo_maps(
                image, 
                word_bboxes,
                map_scale=map_scale
            )

            # 결과 저장
            total["images"].append(image)
            total["vertices"].append(vertices)
            total["labels"].append(labels)
            total["word_bboxes"].append(word_bboxes)
            total["roi_masks"].append(roi_mask)
            total["score_maps"].append(score_map)  # EAST score map 저장
            total["geo_maps"].append(geo_map)      # EAST geo map 저장
            
        except Exception as e:
            print(f"Error processing image {image_fpath}: {str(e)}")
            continue

    print(f"Save path >> {pkl_dir}")
    with open(pkl_dir, 'wb') as fw:
        pickle.dump(total, fw)

if __name__ == "__main__":
    preprocessing(
        root_dir='/data/ephemeral/home/code/data',
        split="train",
        num=2,
        image_size=2048,
        crop_size=1024,
        ignore_under_threshold=10,
        drop_under_threshold=1,
        map_scale=0.5 
    )