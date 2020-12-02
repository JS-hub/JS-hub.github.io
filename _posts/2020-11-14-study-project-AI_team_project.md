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
- #### 5.Appendix
  - #### 5-1 전처리 시행착오 

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

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림1.  전체 사진의 가로길이 분포**

<img src = "https://JS-hub.github.io\assets\img\study\all_height.png" >

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림2. 전체 사진의 세로길이 분포**

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

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림3. 얼굴 사진의 가로길이 분포**
<img src = "https://JS-hub.github.io\assets\img\study\face_height.png" >

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림4. 얼굴 사진의 세로길이 분포**

얼굴사진 크기의 분포를 본 결과 가로는 220에 세로는 200을 중심으로 모여있었습니다. 하지만 전체 사진으로 학습한 모델과 비교하기 위해 **case1)**과 같이 (400,300)으로리사이즈 해주기로 했습니다. 
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

## 3.Model 
---
사용할 모델은 CNN을 기반으로 하는 VGGNET에 Batch Normalization을 추가하였으며 손실함수는 Cross-entropy를 사용하였고 최적화함수는 Adamoptimizer를 사용하였습니다. Model을 구성할 것입니다. 모델 구현에 앞서 모델이 어떤 방식으로 훈련하는지 설명드리겠습니다.
## CNN(Convolutional Neural network)
기존의 DNN에서는 이미지를 1D 형태로 펴주었는데요. 이런 방법은 데이터를 변경하는 과정에서 공간적인 정보의 손실이 생깁니다. 그래서 기존 신경망이 특징을 추출하고 학습하는데 있어서 한계가 발생하게 됩니다. 이러한 단점을 보완하여 이미지의 공간정보를 유지한채로 학습하는 모델이 CNN입니다.
![image](https://JS-hub.github.io\assets\img\study\CNN.png)
**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림5. CNN의 구조**

CNN은 크게 FeatureExtraction layer와 Classificaion layer로 나뉩니다. FeatureExtraction layer의 경우 Convolution layer와 Pooling Layer로 이루어져 있으며 Classification layer는 Fully-Connected layer(완전 연결 계층)으로 이루어져있습니다. 
### Convolution
![image](https://JS-hub.github.io\assets\img\study\convolution.gif)
**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림6. Convolution 예시 이미지**

그림6에서 보면 `3X3`의 노란색 필터가 이미지에 적용되면서 `Feature`를 뽑아내는 것을 알 수 있습니다. 필터에 각 가중치가 할당되고 이 할당된 값(**그림6**의 빨간 숫자)을 이미지에 곱연산하여 특징을 추출하는 것입니다. 이렇게 생성된 이미지를 'Feature map'이라고 합니다. **그림 6**에서 필터가 오른쪽으로 한칸씩 아래로도 한칸씩 움직이면서 진행됩니다. 이처럼 지정된 간격으로 필터가 이미지를 순회하는 간격을 `stride`라고 하며 이때는 `stride`가 1입니다.

### Channel 
<img src = "https://pbs.twimg.com/media/DqaynZSVAAASsgl.jpg">

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림7. color 이미지**


**그림7**과 같은 color이미지는 Red channel,Blue Channel,Green Channel이 합쳐진 3채널 이미지 입니다. 즉 하나의 color 이미지는 3개의 채널로 이루어져있습니다. 흑백사진의 경우는 1채널입니다.

### Padding 
**그림6**을 보면 특징을 추출하기 전의 이미지는 `5x5`의 크기였지만 `3x3`의 필터를 통해 추출한 'Feature map'의 크기는 `3x3`이 되었습니다. 이처럼 출력 데이터가 입력데이터 보다 작아지는데요. 이를 방지하는 방법이 `padding`입니다.

<img src = "https://JS-hub.github.io\assets\img\study\padding.png">

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림8. Padding 예시 이미지**

**그림8**과 같이 0으로 둘러싸는 `padding`을 `zero padding`이라고 합니다. 이미지의 모든 방향으로 한 픽셀만큼 0을 채워주면 `3X3`필터를 통과해도 `Feature map`의 크기가 줄어들지 않습니다. 만약 `5X5`크기의 필터를 사용하게 된다면 모든 방향으로 두 픽셀만큼 채워줘야겠죠? 

### Pooling
이미지의 크기를 계속 유지하면서 레이어를 늘리면 연산량이 엄청 늘어날 것입니다. 그래서 크기를 줄이면서 `Feature map`의 대표값만 뽑아내는 것을 `pooling`이라고 합니다. 
<img src = "https://JS-hub.github.io\assets\img\study\pooling.png">

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림9. Max Pooling 예시 이미지**

**그림9**는 `Maxpooling`을 보여주는 이미지 입니다. `2X2`크기의 필터에 `stride`를 2로 적용시켰을 때 필터에 영역에 해당하는 값 중에 가장 큰 값만은 뽑아 내는 것입니다. 평균 값을 뽑아내는 경우 `Average pooling` 이라고 합니다. 이미지 분류에서는 주로 `MaxPooling`을 사용합니다.

### Batch Normalization

### VGGNet
VGGNet은 2014년 ImageNet이라는 1000개의 이미지를 구별하는 대회에서 좋은 성적을 낸 모델입니다. 

<img src = "https://JS-hub.github.io\assets\img\study\VGGNet.png">

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;그림10. VGGNet layer 구성표**

저희의 경우 13layer모델을 기준으로 만들었습니다. input이미지의 크기가 400x300으로 더 크지만 Convolution layer에서 Filter의 크기가 작을 수록 특징을 잡기에 좋다는 점 때문에 Convolution layer는 그대로 유지하였습니다. 하지만 구별해야하는 것이 1000개인 반면 저희 모델은 2개만 구분하면 됐기 때문에 classificaion layer의 노드 수를 256으로 조정했습니다.

### Cross-entropy

### Adam Optimmizer

## 모델구현 
```python
import tensorflow.compat.v1 as tf # 텐서플로우 버전 1을 사용
tf.disable_v2_behavior() 

X = tf.placeholder(tf.float32, [None,400,300,3])
Y = tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

W1 = tf.Variable(tf.random_normal([3,3,3,64],stddev=0.01)) 
L1 = tf.nn.conv2d(X,W1, strides=[1,1,1,1],padding = 'SAME') 
L1 = tf.layers.batch_normalization(L1, training = is_training)
L1 = tf.nn.relu(L1)
# 400 300  64 

W2 = tf.Variable(tf.random_normal([3,3,64,64],stddev=0.01))
L2 = tf.nn.conv2d(L1,W2, strides =[1,1,1,1],padding = 'SAME') 
L2 = tf.layers.batch_normalization(L2, training = is_training)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 200 150 64

W3 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
L3 = tf.nn.conv2d(L2,W3, strides =[1,1,1,1],padding = 'SAME') 
L3 = tf.layers.batch_normalization(L3, training = is_training)
L3 = tf.nn.relu(L3)

W4 = tf.Variable(tf.random_normal([3,3,128,128],stddev=0.01)) 
L4 = tf.nn.conv2d(L3,W4, strides=[1,1,1,1],padding = 'SAME') 
L = tf.layers.batch_normalization(L4, training = is_training)
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
#100 75 128

W5 = tf.Variable(tf.random_normal([3,3,128,256],stddev=0.01))
L5 = tf.nn.conv2d(L4,W5, strides =[1,1,1,1],padding = 'SAME') 
L5 = tf.layers.batch_normalization(L5, training = is_training)
L5 = tf.nn.relu(L5)

W6 = tf.Variable(tf.random_normal([3,3,256,256],stddev=0.01))
L6 = tf.nn.conv2d(L5,W6, strides =[1,1,1,1],padding = 'SAME') 
L6 = tf.layers.batch_normalization(L6, training = is_training)
L6 = tf.nn.relu(L6)
L6 = tf.nn.max_pool(L6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 50 38 256 

W7 = tf.Variable(tf.random_normal([3,3,256,512],stddev=0.01))
L7 = tf.nn.conv2d(L6,W7, strides =[1,1,1,1],padding = 'SAME') 
L7 = tf.layers.batch_normalization(L7, training = is_training)
L7 = tf.nn.relu(L7)

W8 = tf.Variable(tf.random_normal([3,3,512,512],stddev=0.01))
L8 = tf.nn.conv2d(L7,W8, strides =[1,1,1,1],padding = 'SAME') 
L8 = tf.layers.batch_normalization(L8, training = is_training)
L8 = tf.nn.relu(L8)
L8 = tf.nn.max_pool(L8,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 25 19 512

W9 = tf.Variable(tf.random_normal([3,3,512,512],stddev=0.01))
L9 = tf.nn.conv2d(L8,W9, strides =[1,1,1,1],padding = 'SAME') 
L9 = tf.layers.batch_normalization(L9, training = is_training)
L9 = tf.nn.relu(L9)

W10 = tf.Variable(tf.random_normal([3,3,512,512],stddev=0.01))
L10 = tf.nn.conv2d(L9,W10, strides =[1,1,1,1],padding = 'SAME') 
L10 = tf.layers.batch_normalization(L10, training = is_training)
L10 = tf.nn.relu(L10)
L10 = tf.nn.max_pool(L10,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 13 10 512

W11 = tf.Variable(tf.random_normal([13*10*512,256],stddev=0.01))
L11 = tf.reshape(L10,[-1,13*10*512])
L11 = tf.matmul(L11,W11)
L11 = tf.nn.relu(L11)
L11 = tf.nn.dropout(L11, keep_prob)

W12 = tf.Variable(tf.random_normal([256,256],stddev=0.01))
L12 = tf.matmul(L11,W12)
L12 = tf.nn.relu(L12)
L12 = tf.nn.dropout(L12, keep_prob)

W13 = tf.Variable(tf.random_normal([256,2],stddev=0.01))
model = tf.matmul(L12,W13)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
```

## Evaluate 
---

<br>

## Disscussion
---

## Appendix 
---
### 전처리 시행착오: 워터마크 제거
이미지 크롤링을 통해 데이터를 수집하는 과정에서 크롤링한 사진 중 다수의 파일에 저작권 표시를 위한 워터마크가 삽이되어 있었습니다. 워터마크가 있으면 불필요한 Feature가 검출되고 이것이 모델이 학습하는데에 문제를 줄 것이라고 판단하였고 워터마크를 지우거나 워터마크가 심한 파일을 삭제하기로 하였습니다. 
#### 1. 이미지 임계처리 
- 이미지를 흑/백으로 분류하여 처리하는 이진화를 이용해 워터마크를 제거하는 방법입니다. 이때 기준이 되는 임계값을 어떻게 결정할 것인지가 중요합니다. 임계값보다 크면 백, 작으면 흑으로 처리됩니다. 
<br>
- Simple thresholding: 사용자가 고정된 임계값을 결정하고 그 결과를 보여주는 단순한 형태입니다.

``` python
img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/femaleTeenager1.jpg',0)

# ret에는 임계값이 할당되며 thresh에는 thresholding된 이미지가 할당됩니다.
ret, thresh1 = cv2.threshold(img,150,255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img,150,255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img,150,255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img,150,255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img,150,255, cv2.THRESH_TOZERO_INV)

titles =['Original','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()
```

<img src = "https://JS-hub.github.io\assets\img\study\simple_th.png" >


- Adaptive thresholding: 이미지의 작은 영역별로 thresholding 하는 방법입니다. 정해준 임의의 값이 아닌 bookSize*bookSize에서 구할 수 있기 때문에 더 정확하게 threshold를 적용할 수 있습니다. 

```python
img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/femaleTeenager1.jpg',0)

ret, th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
cv2.imwrite("D:\\test1\\nonwatermarkTest_6.jpg",th1)
cv2.imwrite("D:\\test1\\nonwatermarkTest_7.jpg",th2)
cv2.imwrite("D:\\test1\\nonwatermarkTest_8.jpg",th3)

titles = ['Original','Global','Mean','Gaussian']

images = [img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
```
<img src = "https://JS-hub.github.io\assets\img\study\adaptive_th.png" >

#### 한계
Binary와 Global에서 변환된 이미지를 보면 Thresholding으로 워터마크를 제거할 수 잇지만 원본 이미지가 손상된다는 것을 알 수있었습니다. 따라서 Thresholding을 통한 워터마크 제거는 적합하지 않다고 판단했습니다.
