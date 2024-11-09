# 다국어 영수증 OCR

## 🥇 팀 구성원

<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kimsuckhyun">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004010%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김석현</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/kupulau">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003808%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>황지은</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/lexxsh">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003955%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이상혁</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/june21a">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003793%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>박준일</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/glasshong">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004034%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>홍유리</b></sub><br />
      </a>
    </td>
  </tr>
</table>
</div>

<br />

## 🗒️ 프로젝트 개요

카메라로 영수증을 인식할 경우 자동으로 영수증 내용이 입력되는 어플리케이션이 있습니다. 이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

<br />

## 📅 프로젝트 일정

프로젝트 전체 일정

- 2024.10.28 (월) 10:00 ~ 2024.11.7 (목) 19:00

### 수정하기

![image](https://github.com/user-attachments/assets/5117d3e0-3af3-4f68-b029-8059dbcaa3da)

## 💻 개발 환경

```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Wandb, Notion
```

## 🏆 프로젝트 결과

## ✏️ Wrap-Up Report

- 프로젝트의 전반적인 내용은 아래 랩업 리포트를 참고 바랍니다.
- [Wrap-Up Report]

## 📁 데이터셋 구조

```
📦data
 ┣ 📂chinese_receipt
 ┃ ┣ 📂img
 ┃ ┣ 📂ufo
 ┃ ┗ ...
 ┃
 ┣ 📂japanese_receipt
 ┃ ┣ 📂img
 ┃ ┣ 📂ufo
 ┃ ┗ ...
 ┃
 ┣ 📂thai_receipt
 ┃ ┣ 📂img
 ┃ ┣ 📂ufo
 ┃ ┗ ...
 ┃
 ┣ 📂vietnamese_receipt
 ┃ ┣ 📂img
 ┃ ┣ 📂ufo
 ┃ ┗ ...
```

- 기본적으로 제공된 학습에 사용할 이미지는 총 400장이며 각 언어(중국어, 일본어, 태국어, 베트남어)로 나뉘어져 구성되어 있습니다.
- 제공되는 이미지 데이터셋은 UFO(Upstage Format OCR) 형태로 이루어져 있습니다.

<br />

## 📁 프로젝트 구조

```

📦level2-datacentric-cv-05
 ┣ 📂utils
 ┃ ┣ 📜COCO_2_UFO.py                # COCO -> UFO format 변경 코드
 ┃ ┣ 📜UFO_2_COCO.py                # UFO -> COCO format 변경 코드
 ┃ ┣ 📜align_image_name_format.py   # image 이름 포맷 정리 도구
 ┃ ┣ 📜merge_json.py                # 데이터셋 병합 도구
 ┃ ┣ 📜inference_custom.py          # 추론(custom set) 도구
 ┃ ┣ 📜visualize_test.py            # 이미지 시각화(test set) 도구
 ┃ ┣ 📜visualize_custom.py          # 이미지 시각화(custom set) 도구
 ┃ ┣ 📜pickle_preprocessing.py      # 데이터 전처리 유틸리티(pickle)
 ┃ ┣ 📜evaluation_util.py           # val을 평가할 eval을 f1 score로 설정
 ┃ ┣ 📜emsemble.py                  # 앙상블 코드
 ┃ ┗ 📜train_val_split.py           # 데이터 train, val set 분리 유틸리티
 ┃
 ┣ 📜train.py                       # 변경된 dataset에 따라 수정
 ┣ 📜inference.py                   # 저장된 checkpoint에 따라 수정
 ┣ 📜dataset.py                     # pickle 파일로 학습할 수 있도록 변경
 ┣ 📜deteval.py
 ┣ 📜model.py
 ┣ 📜loss.py
 ┣ 📜east_dataset.py
 ┣ 📜requirements.txt
 ┣ 📜datect.py
 ┗ 📜README.md


```

<br />

## 🧱 Structure

</details>

<br />
