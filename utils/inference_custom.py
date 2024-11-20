import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import sys
sys.path.append('..')

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_path', required=True, help='Path to the model checkpoint file')
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    checkpoint = torch.load(ckpt_fpath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    images = []
    
    # 폴더 내의 모든 이미지 파일을 찾음
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:  # 지원할 이미지 확장자들
        image_files.extend(glob(osp.join(data_dir, ext)))
        image_files.extend(glob(osp.join(data_dir, ext.upper())))  # 대문자 확장자도 포함
    
    # 각 이미지에 대해 처리
    for image_fpath in tqdm(image_files):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, args.model_path, args.data_dir, args.input_size,  # 변경된 부분
                              args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = 'output.json'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
