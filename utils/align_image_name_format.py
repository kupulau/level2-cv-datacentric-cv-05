import os
import json
import argparse



def parse_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_dir', type=str, default='/data/ephemeral/home/code/additional_data/vietnamese_receipt/img/train')
    parser.add_argument('--json_file', type=str, default='/data/ephemeral/home/code/additional_data/vietnamese_receipt/ufo/Custom_viet_batch_1_UFO.json')
    parser.add_argument('--lang', type=str, default='vi')
    parser.add_argument('--version', type=str, default='V1.0')
    
    args = parser.parse_args()
    return args


def get_fname_format(lang, idx, version):
    return f"extractor.{lang}.addition_{version}_page{idx}.jpg"


def rename_files_in_directory(directory_path, json_path, lang, version):
    try:
        # 해당하는 json파일 가져오기
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        
        new_dict = {'images': {}}
        # 디렉토리에 있는 파일 목록 가져오기
        for idx, filename in enumerate(json_dict['images']):
            # 파일의 전체 경로 생성
            old_file_path = os.path.join(directory_path, filename)
            new_name = get_fname_format(lang, idx, version)
            
            new_file_path = os.path.join(directory_path, new_name)
            os.rename(old_file_path, new_file_path)
            new_dict['images'][new_name] = json_dict['images'][filename]
            
            print(f"Renamed: {old_file_path} -> {new_file_path}")
        
        with open(json_path, 'w') as f:
            json.dump(new_dict, f)
        print("모든 파일의 이름을 변경했습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")


def main():
    args = parse_argument()
    img_dir = args.img_dir
    json_file = args.json_file
    lang = args.lang
    version = args.version
    
    rename_files_in_directory(img_dir, json_file, lang, version)

if __name__ == "__main__":
    main()