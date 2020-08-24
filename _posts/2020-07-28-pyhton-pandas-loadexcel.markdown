---  
layout: post  
title: "판다스의 기본"  
subtitle: "pandas"  
categories:  python  
tags: pandas 판다스 정리
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
#### 시리즈 만들기
``` python
import pandas as pd

pd.Series(['a','b','c','d'])
pd.Series({0:'a',1:'b',2:'c',3:'d'})

# 0    a
# 1    b
# 2    c
# 3    d
# dtype: object
```


#### 시리즈 매서드
Series.value : 해당 시리즈 객체의 값들만 반환<br>
Series.index : 해당 시리지의 index만 반환<br>
Series[index] :  배열에서 값을 선택할 때 사용 <br>
Series[condition] : 조건문에 해당하는 Index와 값을 반환 <br>
ex)
``` python
import pandas as pd

a = pd.Series(['a','b','c','d'])
a[a == 'a']

# 0    a
# dtype: object
```
<br>
#### 데이터 프레임 만들기

``` python
import pandas as pd

df = pd.DataFrame({'국어':{'철수': 100,'영희': 90,'지훈': 80},
                   '수학':{'철수': 90,'영희': 80,'지훈': 90},
                   '영어':{'철수': 70,'영희': 90,'지훈': 50}})

#        국어	수학	영어
#철수	100	90	70
#영희	90	80	90
#지훈	80	90	50
```
<br>
``` python
df = pd.DataFrame([[100, 90, 70],
                    [90, 80, 90],
                    [80, 90, 50]], 
                   index = ['철수','영희','지훈'],
                   columns =['국어','수학','영어'] )

#        국어	수학	영어
#철수	100	90	70
#영희	90	80	90
#지훈	80	90	50
```
#### 데이터 프레임 매서드
시리즈의 매서드는 데이터 프레임에도 적용할 수 있다.<br>
Dataframe.T : 데이터 프레임의 열과 행을 바꿔줌<br>
Dataframe.columns : 데이터 프레임의 열을 반환<br>
<br>
#### 데이터 행,열 추가

``` python
#데이터 열 추가하기

df['과학'] = [100,90,80]

# 	국어	수학	영어	과학
# 철수	100	90	70	100
# 영희	90	80	90	90
# 지훈	80	90	50	80
```

``` python
#데이터 행 추가하기

df2 = pd.DataFrame([[100, 50, 100, 60]],
                    index = ['지수'],
                    columns =['국어','수학','영어','과학'])

pd.concat([df,df2],axis=0)
df = df.append(df2, ignore_index= False)

# 	국어	수학	영어	과학
# 철수	100	90	70	100
# 영희	90	80	90	90
# 지훈	80	90	50	80
# 지수	100	50	100	60
```
#### 데이터 행,열 제거, 추출
``` python
#데이터 행, 열 제거
df = df.drop(['과학'],axis =1 )
df = df.drop(['지수'],axis =0 )
```
행을 제거 할때 axis =0 ,열을 제거할 때 axis =1 

#### 데이터 프레임의 행별 평균을 구하고 평균 순서대로 정렬하기

```python
# df 
# 	국어	수학	영어	과학
# 철수	100	90	70	100
# 영희	90	80	90	90
# 지훈	80	90	50	80

df['평균']=df.sum(axis =1)/4.0
df.sort_values(['평균'],ascending= False)

#       국어	수학	영어	과학	평균
# 철수	100	90	70	100	90.0
# 영희	90	80	90	90	87.5
# 지수	100	50	100	60	77.5
# 지훈	80	90	50	80	75.0
```

#### 데이터 프레임 Apply 적용해 보기
```python
# 위 데이터에서 평균을 기준으로 등급 부여하기
def grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    else:
        return 'F'

df['등급'] = df['평균'].apply(grade)


#       국어	수학	영어	과학	평균	등급
# 철수	100	90	70	100	90.0	A
# 영희	90	80	90	90	87.5	B
# 지훈	80	90	50	80	75.0	C
# 지수	100	50	100	60	77.5	C
```
#### 데이터 프레임에서 중복 데이터 제거와 데이터 그룹 만들기

```python
df = pd.DataFrame({ '이름':{0:'철수',1:'영희',2:'지훈',3:'수현',4:'예지',5:'민지',6:'지훈'},
                    '학과':{0:'컴퓨터공학과',1:'물리학과',2:'수학과',3:'철학과',4:'수학과',5:'영어영문학과',6:'수학과'},
                    '입학연도':{0:2020,1:2020,2:2019,3:2020,4:2020,5:2020,6:2019}
})
#       이름	학과	      입학연도
# 0	철수	컴퓨터공학과	2020
# 1	영희	물리학과	2020
# 2	지훈	수학과	        2019
# 3	수현	철학과	        2020
# 4	예지	수학과	        2020
# 5	민지	영어영문학과	2020
# 6	지훈	수학과	        2019

df = df.drop_duplicates(keep='first')

#       이름	학과	      입학연도
# 0	철수	컴퓨터공학과	2020
# 1	영희	물리학과	2020
# 2	지훈	수학과	        2019
# 3	수현	철학과	        2020
# 4	예지	수학과	        2020
# 5	민지	영어영문학과	2020

major = df.groupby(['학과'])
major.groups
# {'물리학과': Int64Index([1], dtype='int64'),
#  '수학과': Int64Index([2, 4], dtype='int64'),
#  '영어영문학과': Int64Index([5], dtype='int64'),
#  '철학과': Int64Index([3], dtype='int64'),
#  '컴퓨터공학과': Int64Index([0], dtype='int64')}

major.size
# 학과
# 물리학과      1
# 수학과       2
# 영어영문학과    1
# 철학과       1
# 컴퓨터공학과    1
# dtype: int64

for name, group in major:
    print(name+" : " +str(len(group)))
    print(group)
    print()

# 물리학과 : 1
#    이름    학과  입학년도
# 1  영희  물리학과  2020

# 수학과 : 2
#    이름   학과  입학년도
# 2  지훈  수학과  2019
# 4  예지  수학과  2020

# 영어영문학과 : 1
#    이름      학과  입학년도
# 5  민지  영어영문학과  2020

# 철학과 : 1
#    이름   학과  입학년도
# 3  수현  철학과  2020

# 컴퓨터공학과 : 1
#    이름      학과  입학년도
# 0  철수  컴퓨터공학과  2020
```