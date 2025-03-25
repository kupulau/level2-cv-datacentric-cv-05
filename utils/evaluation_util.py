import os
import os.path as osp
import zipfile
import re
import json


def make_default_folder_for_evaluate(evaluation_dir):
    folder_name = osp.join(evaluation_dir, "gt")
    os.makedirs(folder_name, exist_ok=True)
    folder_name = osp.join(evaluation_dir, "result")
    os.makedirs(folder_name, exist_ok=True)


def write_result_txt(evaluation_dir, mode, bboxes, idx):
    file_path = osp.join(evaluation_dir, mode, f"img_{idx}.txt")
    bboxes = bboxes.astype(int)
    
    # print(mode)
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            line = ",".join(map(str, bbox))
            if mode == "gt":
                line += ",concierge@L3"
            
            # print(line)
            f.write(line + "\n")
    
    return file_path
    
    
def zip_from_list(evaluation_dir, mode, file_pathes):
    zip_path=  osp.join(evaluation_dir, f"{mode}.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in file_pathes:
            file_name = os.path.basename(file)
            zipf.write(file, arcname=file_name)
    
    return zip_path


def decode_result(stdout_data):
    """
    byte 문자열로 얻어진 TedEval의 output에서 recall, precision, hmean을 key로 가지는 dict만 뽑아오는 코드
    만약 해당 함수에서 오류가 난 경우 stderr이 얻어져서 TedEval이 제대로 수행되지 않은 것
    """
    decoded_str = stdout_data.decode('utf-8')
    matches = re.findall(r"(\{.*?\})", decoded_str)
    json_dict = json.loads(matches[2])
    return json_dict
    