---  
layout: post  
title: " 2020-2 딥러닝 팀 프로젝트(Option1) "  
subtitle: "project"  
categories:  study
tags: project 딥러닝 AI 
comments: false  
---  
# 딥러닝을 통한 남녀 분류 모델 만들기
---

## Index
- #### 1.Introduction 
- #### 2.Datasets
  - #### 2-1. 이미지 크롤링
  - #### 2-2.이미지 리사이즈
  - #### 2-3.라벨링 
  - #### 2-4.학습/테스트 데이터셋 분리
- #### 3.Model
  - #### 3-1 CNN(Convolutional Neural Network)
  - #### 3-2 모델구현
- #### 4.Evaluation & Disscussion
  - #### 4-1 모델평가 

<br>

## 1.Introduction
---
 저희 팀은 직접 데이터를 만들고 테스트 할 수 있으면 좋겠다고 생각했습니다. 그래서 남녀 분류 모델을 만들기로 계획을 했고 모델을 다 학습시킨 뒤에는 저희 사진을 모델에 넣으면 어떻게 분류할지 확인할 것입니다.
<br>

## 2.Datasets 
---
남녀 구분 모델에 필요한 데이터 셋은 남자와 여자 사진입니다. kaggle에서 이미지 데이터 셋을 구할 수 있었지만 조원들끼리 직접 이미지를 수집하고 처리하여 데이터 셋을 만들고자 하였습니다. 또한 최대한 다양한 인종과 나이대에 대해 적용시킬 수 있게 조원간에 이미지 수집 분야를 나눴습니다. 
```
김덕성 : 초등 ~ 청소년
김아영 : 아이돌 배우 (국내 위주)
박재선 : (외국) 배우 가수
장진웅 : 노인 남녀
```


### 이미지 크롤링
이미지는 구글 크롤링을 통하여 수집을 하였고 크롤링에 필요한 툴은 <a href="https://github.com/Joeclinton1/google-images-download.git" target="_blank">https://github.com/Joeclinton1/google-images-download.git</a>에서 클론하여 실행하였습니다. 개발환경은 구글코랩이며 크롤링하는 코드는 다음과 같습니다.

```python
!git clone https://github.com/Joeclinton1/google-images-download.git

cd google-images-download

from google_images_download import google_images_download  

response = google_images_download.googleimagesdownload()   
arguments = {"keywords":"male,female","limit":100,"print_urls":False}  
# 키워드에 들어가는 단어로 검색된 이미지로 limit에 설정된 수만큼 크롤링을 합니다.
paths = response.download(arguments)  

```
크롤링된 데이터를 확인해보고 사람이 아닌 사진과 같이 학습에 적합하지 못한 사진들은 직접 수작업으로 제거했습니다. 

<br>

### 이미지 리사이즈
#### Case1) 전신 사진을 포함한 모든 사람 사진
모델에 이미지를 집어넣기 위해서 모두 같은 크기의 사진으로 설정해주어야 합니다. 따라서 이미지크기 분포를 확인해야 합니다. 분포를 확인하기위한 코드는 다음과 같습니다.

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

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림1.  전체 사진의 가로길이 분포**

<img src = "https://JS-hub.github.io\assets\img\study\all_height.png" >

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림2. 전체 사진의 세로길이 분포**

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
            #여성사진의 경우 female_number로 rename해줍니다.
                cv2.imwrite((os.path.join(total_path , 'female_')+str(i)+'.'+imgpath.split('.')[-1]), img)
                i+=1
            else:
            #남성사진의 경우 male_number로 rename해줍니다.
                cv2.imwrite((os.path.join(total_path , 'male_')+str(j)+'.'+imgpath.split('.')[-1]), img)
                j+=1
```

### Case2) 얼굴사진만을 포함한 사람사진
얼굴 사진만을 다시 데이터셋으로 구성한 이유는 **case1)** 에서 모든 사진을 포함한 모델을 학습시켰을 때 정확도가 잘 나오지 않아서 조원끼리 상의해 본 결과 모은 사진이 너무 다양하여 사진의 Feature를 제대로 학습하지 못하였다고 판단하였습니다. 그래서 얼굴 사진으로 통일하여 다시 학습시키기 위하여 얼굴 사진만으로 이루어진 데이터셋을 만들었습니다. 
<img src = "https://JS-hub.github.io\assets\img\study\face_width.png" >

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림3. 얼굴 사진의 가로길이 분포**
<img src = "https://JS-hub.github.io\assets\img\study\face_height.png" >

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림4. 얼굴 사진의 세로길이 분포**

얼굴사진 크기의 분포를 본 결과 가로는 220에 세로는 200을 중심으로 모여있었습니다. 하지만 전체 사진으로 학습한 모델과 비교하기 위해 **case1)** 과 같이 (400,300)으로리사이즈 해주기로 했습니다. 
<br>

### 사진 라벨링
지도학습을 하기 때문에 각 사진이 남성인지 여성인지 라벨링을 해주어야 합니다. 라벨링은 원-핫 인코딩을 사용했습니다.
#### 원-핫 인코딩
컴퓨터는 숫자를 잘 처리 할 수 있습니다. 이를 위해 문자를 숫자로 바꾸는 여러가지 기법들이 있습니다. 이러한 기법들 중 **원-핫 인코딩(One-Hot Encoding)**은 단어를 표현하는 가장 기본적인 표현 방법입니다.

원-핫 인코딩이란 표현하고자하는 라벨에 총 단어 수만큼의 길이의 [0]을 만들고 각 단어에 해당하는 고유한 정수 인덱스에만 1을 부여하여 만드는 것입니다. 예를 들어 우리의 경우 남자는 0,여자는 1이라는 정수 인덱스를 부여하면 단어 수는 두 개이기 때문에 [0,0]을 만들고 남자의 경우 [1,0] 여자의 경우[0,1]이 되는 것입니다.

```python
#원-핫 인코딩 라벨 생성
def one_hot_encoding(index, total): 
       one_hot_vector = [0]*(len(total)) 
       one_hot_vector[index]=1 
       return one_hot_vector 

classes = ['male','female']
class_table = {cls:one_hot_encoding(i,classes) for i,cls in enumerate(classes)}
class_table

# 딕셔너리형태로 라벨과 이미지를 묶어서 저장해줍니다.
imgs = (os.listdir(total_path))
dataset = {cls : [[],[]] for cls in classes}
for img in imgs:
    cls = img.split('_')[0] #
    imgpath = os.path.join(total_path, img)
    img = cv2.imread(imgpath)
    dataset[cls][0].append(img)
    dataset[cls][1].append(class_table[cls])

dataset = {key : [np.array(value[0]), np.array(value[1])] for key, value in dataset.items()} 
```
### 학습/테스트 데이터셋 분리
위의 과정까지해서 이미지와 라벨을 묶었습니다. 이제 다음으로는 학습 데이터셋과 테스트용 데이터셋을 분리해야 합니다. 그 이유는 과적합 때문입니다. 과적합이란 모델이 훈련 데이터셋에 편향되어 훈련되지 않은 데이터에 대해 잘맞추지 못하는 것을 말합니다. 따라서 잘 학습되었는지 확인하기 위해 테스트 데이터셋을 따로 분리해줍니다. 
```python
output = [[],[],[],[]] 
total_tr, total_te = 0, 0 
for key, value in dataset.items(): 
        split_id = int(len(dataset[key][0]) * 0.7 ) 
        X = dataset[key][0]
        Y = dataset[key][1]
        # split_id에 해당하는 인덱스를 기준으로 분리해줍니다.
        X_train, X_test = np.split(X, [split_id]) 
        Y_train, Y_test = np.split(Y, [split_id])
        output[0].append(X_train) 
        output[1].append(Y_train)
        output[2].append(X_test)
        output[3].append(Y_test)
        n_tr, n_te = split_id, len(dataset[key][1]) - split_id 
        total_tr += n_tr 
        total_te += n_te 
``` 
이렇게 데이터셋을 분리하면 데이터를 섞어주는 작업이 필요합니다. 지금 데이터 상태로 학습시키면 남자사진만 연속적으로 보여주기 때문에 데이터를 섞어줄 필요가 있습니다. 데이터의 순서를 섞어주는 코드는 다음과 같습니다.

```python
tr_id = np.random.choice(total_tr, total_tr, replace=False) 
te_id = np.random.choice(total_te, total_te, replace=False)
for i in range(len(output)): 
    tmp = np.concatenate(output[i],axis=0) 
    output[i] = tmp[tr_id] if i < 2 else tmp[te_id] 

X_train, Y_train = output[0], output[1] 
X_test, Y_test = output[2], output[3]
```
여기까지 전처리를 마치고 이제 모델에 집어넣을 데이터 준비가 끝났습니다.

## Model 
---
사용할 모델은 CNN을 기반으로 하는 VGGNET에 Batch Normalization을 추가하였으며 손실함수는 Cross-entropy를 사용하였고 최적화함수는 Adamoptimizer를 사용하였습니다. Model을 구성할 것입니다. 모델 구현에 앞서 모델이 어떤 방식으로 훈련하는지 설명드리겠습니다.


## Evaluate 

<br>

## Disscussion
