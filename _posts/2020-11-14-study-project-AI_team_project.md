---  
layout: post  
title: " 2020-2 딥러닝 팀 프로젝트(Option1) "  
subtitle: "project"  
categories:  study
tags: project 딥러닝 AI 
comments: false  
---  
# 딥러닝을 통한 남녀 분류 모델 만들기
<br>

## Index
#### 1.Introduction 
#### 2.Datasets
#### 3.Methods
#### 4.Model & Evaluation

<br>

## 1.Introduction
 저희 팀은 직접 데이터를 만들고 테스트 할 수 있으면 좋겠다고 생각했습니다. 그래서 남녀 분류 모델을 만들기로 계획을 했고 모델을 다 훈련시킨 뒤에는 저희 사진을 모델에 넣으면 어떻게 분류할지 확인할 것입니다.
<br>

## 2.Datasets 
남녀 구분 모델에 필요한 데이터 셋은 남자와 여자 사진입니다. kaggle에서 이미 있는 이미지 데이터 셋을 구할 수 있었지만 조원들끼리 직접 이미지를 모아 불필요한 이미지를 따로 처리하여 데이터 셋을 만들고자 하였습니다. 최대한 다양한 인종과 나이대에 대해 적용시킬 수 있게 조원간에 이미지 수집분야를 나눴습니다. 
```
김덕성 : 초등 ~ 청소년
김아영 : 아이돌 배우 (국내 위주)
박재선 : (외국) 배우 가수
장진웅 : 노인 남녀 (외국 위주) 
```
이미지 수집은 구글 크롤링을 통하여 수집을 하였고 크롤링에 필요한 툴을 <a href="https://github.com/Joeclinton1/google-images-download.git" target="_blank">github</a>에서 다운받아 실행하였습니다. 개발환경은 구글코랩이며 크롤링하는 코드는 다음과 같습니다.
```python
!git clone https://github.com/Joeclinton1/google-images-download.git

cd google-images-download

from google_images_download import google_images_download  

response = google_images_download.googleimagesdownload()   
arguments = {"keywords":"male,female","limit":100,"print_urls":False}  
# 키워드에 들어가는 단어로 검색된 이미지로 limit에 설정된 수만큼 크롤링을 합니다.
paths = response.download(arguments)  

```
크롤링된 데이터에서 학습에 부적절하다고 생각된 사진들은 직접 수작업으로 제거했습니다. 


### Case1) 전신사진을 포함한 모든 사람사진
모델에 이미지를 집어넣기 위해서 모두 같은 크기의 사진으로 설정해주어야 합니다. 또한 코랩에서 사용하기 때문에 램 크기를 고려하여 적절한 크기로 조절해야합니다. 그러기 위해 이미지크기 분포를 확인합니다. 분포를 확인하기위한 코드는 다음과 같습니다.
```python

import os, pickle, cv2
import numpy as np 
import matplotlib.pyplot as plt 

height =[]
width = []
error_list=[]
for path in paths:
    for name in os.listdir(path):
          imgpath = os.path.join(path,name)
          img = cv2.imread(imgpath)
          try: #이미지의 가로 세로를 받아 각가 리스트에 append합니다.
              x,y,channel = img.shape 
              height.append(x)
              width.append(y)
          except: #파일이 안 열리는 경우 안열리는 파일경로를 append합니다.
              error_list.append(imgpath)

plt.hist(height, bins = 100)
plt.hist(width, bins = 100)
```
<img src = "https://JS-hub.github.io\assets\img\study\all_width.png" >

그림1.  전체 사진의 가로길이 분포

<img src = "https://JS-hub.github.io\assets\img\study\all_height.png" >

그림2. 전체 사진의 세로길이 분포
<br> 


분포를 본 결과 가로는 400 세로는 300을 중심으로 모여있는 것을 확인하였고 모든 사진을 (400,300)으로 리사이즈를 하기로 하였습니다. 리사이즈하는 코드는 다음과 같습니다.
```python
i = 0
j = 0
total_path = '/content/drive/Shareddrives/2020_AI_Deep_learning/Dataset/total'
for path in paths:  
    for name in os.listdir(path):
            imgpath = os.path.join(path,name)
            if imgpath in error_list:
                continue
            img = cv2.imread(imgpath)
            height,width,channel = img.shape
            if height>300 and width >400:
                img = cv2.resize(img, (400, 300), interpolation = cv2.INTER_AREA)
            else:
                img = cv2.resize(img, (400, 300), interpolation = cv2.INTER_LINEAR)
            if 'female' in imgpath:
                cv2.imwrite((os.path.join(total_path , 'female_')+str(i)+'.'+imgpath.split('.')[-1]), img)
                i+=1
            else:
                cv2.imwrite((os.path.join(total_path , 'male_')+str(j)+'.'+imgpath.split('.')[-1]), img)
                j+=1
```

### Case2) 얼굴사진만을 포함한 사람사진
<img src = "https://JS-hub.github.io\assets\img\study\face_width.png" >

그림1.  언굴사진의 가로길이 분포

<img src = "https://JS-hub.github.io\assets\img\study\face_height.png" >

그림2. 얼굴사진의 세로길이 분포
<br> 
### 

## Evaluate 

<br>

## Disscussion
