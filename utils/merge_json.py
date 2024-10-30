import json
import glob
import argparse
import os
import sys
import shutil
from tqdm import tqdm


###################################################
########## 다음의 폴더 구조를 필요로 합니다 ##########
# 1번 데이터셋
#  ㄴ chinese_receipt
#     ㄴ img
#        ㄴ train
#        ㄴ test
#          ㄴ img_1.jpg
#          ㄴ img_2.jpg
#          ㄴ ...
#     ㄴ ufo
#        ㄴ train.json
#  ㄴ japanese_receipt
#  ㄴ thai_receipt
#  ㄴ vietnamese_receipt
#
# 2번 데이터셋
# ㄴ ...

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, default = '/data/ephemeral/home/code/merged_dataset')
    parser.add_argument('--mode', type=str, default = 'train')
    
    args = parser.parse_args()
    return args

def get_default_path(target_path, lang, leaf):
    return os.path.join(target_path, f"{lang}_receipt", leaf)


def make_default_folder_tree(target_path, lang_list):
    leafs = ['img/train', 'img/test', 'ufo']
    img_pathes = []
    ufo_pathes = []
    
    for lang in lang_list:
        for leaf in leafs:
            folder_path = get_default_path(target_path, lang, leaf)
            if os.path.exists(folder_path):
                print(f"{folder_path}가 이미 존재합니다.")
            else:
                print(f"{folder_path}를 생성하였습니다.")
                os.makedirs(folder_path, exist_ok=True)
            
            if leaf == 'img':
                img_pathes.append(folder_path)
            else:
                ufo_pathes.append(folder_path)
    
    return img_pathes, ufo_pathes



def merge_img_folder(img_folders, target_path):
    img_types = ['jpeg', 'jpg', 'png']
    
    image_paths = []
    for folder in img_folders:
        for t in img_types:
            image_paths += glob.glob(os.path.join(folder, "*." + t))
    
    cnt = len(image_paths)
    print(f"총 {cnt}개의 이미지를 {target_path}로 복사합니다.")
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(target_path, image_name))
    print(f"총 {cnt}개의 이미지가 {target_path}로 복사되었습니다.\n\n")


def merge_json_files(json_pathes, root_dirs, target_path, mode):
    ufo_json = {'images': {}}
    
    for json_path, root in zip(json_pathes, root_dirs):
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        
        for img_name, img_info in json_dict['images'].items():
            ufo_json['images'][img_name] = img_info
            ufo_json['images'][img_name]['root_dir'] = root
    
    target_json_path = os.path.join(target_path, mode + '.json')
    with open(target_json_path, 'w') as f:
        json.dump(ufo_json, f)
    print(f"target json : {target_json_path}을 생성하였습니다.\n\n")
            

def merge_dataset(target_datasets, lang_list, target_path, mode):
    for lang in lang_list:
        choosed_json_pathes = []
        choosed_img_pathes = []
        
        for dataset in target_datasets:
            json_path = get_default_path(dataset, lang, 'ufo')
            img_path = get_default_path(dataset, lang, 'img/'+mode)
            
            
            if os.path.exists(json_path):
                json_pathes = glob.glob(os.path.join(json_path, '*.json'))
                json_names = [path.split("/")[-1] for path in json_pathes]
                if os.path.exists(img_path):
                    choosed_img_pathes.append(img_path)
                else:
                    print(f"%warning% : annotation에 대응하는 img 폴더가 존재하지 않습니다. {img_path} does not exist")
                
                
                ######################### 입력받기
                default_value = 0
                while True:
                    print(f"dataset : {dataset}, lang : {lang}에 대해서 사용할 json file을 선택해주세요")
                    for idx, name in enumerate(json_names):
                        print(f"{idx} : {name}")
                    json_choice = input("choice (press enter -> defulat 0 or integer): ")
                    
                    if json_choice == "":
                        json_choice = default_value
                        print(f"기본값인 {default_value} : {json_names[default_value]}가 선택되었습니다.\n\n")
                        break
                    
                    if json_choice.isdigit():
                        json_choice = int(json_choice)
                        try:
                            print(f"{json_choice} : {json_names[json_choice]}가 선택되었습니다.\n\n")
                            break
                        except:
                            pass
                    
                    print("엔터를 누르거나 범위에 맞는 숫자를 입력해주세요.\n\n")
            else:
                print(f"dataset {dataset}에 대해 {lang} 폴더가 없으므로 다음 작업으로 넘어갑니다.\n")
            ###########################################################################
            choosed_json_pathes.append(json_pathes[json_choice])
        
        
        # merge json and save to target path
        print(f"다음 json들에 대해 json merge를 시도합니다.")
        for path in choosed_json_pathes:
            print(path)
        print()
        
        img_roots = ['/'.join(path.split("/")[:-2] + ['img']) for path in choosed_json_pathes]
        save_to = get_default_path(target_path, lang, 'ufo')
        merge_json_files(choosed_json_pathes, img_roots, save_to, mode)
        
        
        # merge images and save to target path
        print(f"다음 image folder들에 대해 merge를 시도합니다.")
        for path in choosed_img_pathes:
            print(path)
        print()
        
        save_to = get_default_path(target_path, lang, 'img/' + mode)
        merge_img_folder(choosed_img_pathes, save_to)
            


def main():
    ## args
    args = parse_argument()
    target_path = args.target_path
    mode = args.mode
    
    ## defulat setting
    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
    target_datasets = [
        '/data/ephemeral/home/code/data',
        '/data/ephemeral/home/code/additional_data'
    ]
    
    
    print(f"{target_path}에 데이터셋을 구성하기 위해 폴더 생성을 시도합니다.")
    target_img_pathes, target_ufo_pathes = make_default_folder_tree(target_path, lang_list)
    print("\n\n")
    
    merge_dataset(target_datasets, lang_list, target_path, mode)
    
    

if __name__ == "__main__":
    main()