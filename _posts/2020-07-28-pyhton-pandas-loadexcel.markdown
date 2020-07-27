---  
layout: post  
title: "판다스의 기본"  
subtitle: "pandas "  
categories:  python  
tags: pandas excel 엑셀 
comments: false  
#header-img: img//2020-04-10-review-book-ai-expert-in-one-year-1.jpg  
---  

## 판다스에서 데이터 다루기

#### 엑셀파일 불러오기

파이썬에서 엑셀파일을 불러오는 코드 
``` python
import pandas as pd

df = pd.read_excel('파일.xlsx')
```
### 시리즈 만들기
``` python
import pandas as pd

pd.Series(['a','b','c','d'])
```
실행결과 :
<center>    0 &nbsp; &nbsp; a<br>
            1 &nbsp; &nbsp; b<br>
            2 &nbsp; &nbsp; c<br>
            3 &nbsp; &nbsp; d<br>
           &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  type: object
            </center>
### 시리즈 매서드
Series.value : 해당 시리즈 객체의 값들만 반환<br>
Series.index : 해당 시리지의 index만 반환<br>
Series[index] :  배열에서 값을 선택할 때 사용 <br>
Series[condition] : 조건문에 해당하는 Index와 값을 반환 
ex)




#### 데이터 프레임 만들기

``` python
import pandas as pd

df = pd.DataFrame({'국어':{'철수':100,'영희':90,'지훈':80},
                   '수학':{'철수':90,'영희':80,'지훈':90},
                   '영어':{'철수':70,'영희':90,'지훈':50}})
```
<p style = "text-align":center;><img src =C:\Users\박재선\Desktop\mainblog\JS-hub.github.io\assets\img\pandas\2020-07-28dfimg.png ></p>

#### 데이터 열, 행 추가
