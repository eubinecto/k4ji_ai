# chapter3 선택 분류: 철판 불량 상태 분류 신경망
선택분류는 이진분류의 확장이라 볼 수 있다. 

## 소프트맥스 함수
소프트맥스 함수는 로짓값 벡터를 확률 분포 벡터로 변환해주는 비선형 함수이다. 일반식을 구하면 아래와 같다.

<img src="https://latex.codecogs.com/svg.latex?\;y_{i} = \frac{e^{x_{i}}}{e^{x_{1}}+\cdots +e^{x_{n}}}" />

하지만 이 식은 코드를 통한 계산 과정에서 오류를 일으킬 수 있으므로 아래와 같이 변형된 식을 사용한다.

<img src="https://latex.codecogs.com/svg.latex?\;y_{i} = \frac{e^{x_{i} - x_{k}}}{e^{x_{1}- x_{k}}+\cdots +e^{x_{n}- x_{k}}}" />

#### 소프트맥수 함수 유도
선택 분류를 이진 분류의 확장이라 볼 수 있는 이유는 유도 과정에서 알 수 있다. 이진 분류 유도를 일반화하여 유도한 것이 선택 분류이기 때문이다.
먼저, 이진분류의 유도부터 살펴보자.
수많은 데이터들 중 데이터 X가 선택되었다고 생각하자. 참인 사건을 Y1, 거짓인 사건을 Y2라고 한다면, 아래 식에서 첫 번째 식은 X라는 데이터가 선택되었을 때 사건이 참일 확률이다.

<img src="https://latex.codecogs.com/svg.latex?\;P(Y_{i}\mid X) = \left\{\begin{matrix}
P(Y_{1}\mid X) = \frac{P(X\mid Y_{1})P(Y_{1})}{P(X)} = \frac{P(X\mid Y_{1})P(Y_{1})}{P(X\mid Y_{1})P(Y_{1})+P(X\mid Y_{2})P(Y_{2})}\\ 
P(Y_{2}\mid X) = \frac{P(X\mid Y_{2})P(Y_{2})}{P(X)} = \frac{P(X\mid Y_{2})P(Y_{2})}{P(X\mid Y_{1})P(Y_{1})+P(X\mid Y_{2})P(Y_{2})}
\end{matrix}\right." />

<img src="https://latex.codecogs.com/svg.latex?\;a_{i} = logP(X\mid Y_{i})P(Y_{i})" />
를 정의하여 위의 식에 대입하면  
<img src="https://latex.codecogs.com/svg.latex?\;P(Y_{1}\mid X) = \frac{e^{a_{1}}} {e^{a_{1}}+e^{a_{2}}} = \frac {1} {1+e^{-(a_{1} - a_{2})}}" />
로 표현할 수 있다.

이것을 두 가지 분류에서 n개의 분류로 확장하면 소프트 맥스 함수가 된다.
<img src="https://latex.codecogs.com/svg.latex?\;P(Y_{i}\mid X) = \left\{\begin{matrix}
P(Y_{1}\mid X) = \frac{P(X\mid Y_{1})P(Y_{1})}{P(X)} = \frac{P(X\mid Y_{1})P(Y_{1})}{\sum_{i=1}^{n}P(X\mid Y_{i})P(Y_{i})}\\ 
P(Y_{2}\mid X) = \frac{P(X\mid Y_{2})P(Y_{2})}{P(X)} = \frac{P(X\mid Y_{2})P(Y_{2})}{\sum_{i=1}^{n}P(X\mid Y_{i})P(Y_{i})}\\
\vdots\\
P(Y_{n}\mid X) = \frac{P(X\mid Y_{n})P(Y_{n})}{P(X)} = \frac{P(X\mid Y_{n})P(Y_{n})}{\sum_{i=1}^{n}P(X\mid Y_{i})P(Y_{i})}
\end{matrix}\right." />

시그모이드과 같이  
<img src="https://latex.codecogs.com/svg.latex?\;a_{i} = logP(X\mid Y_{i})P(Y_{i})" />
를 정의하면  
<img src="https://latex.codecogs.com/svg.latex?\;P(Y_{1}\mid X) = \frac{e^{a_{1}}} {e^{a_{1}}+ \cdots +e^{a_{n}}} = \frac {e^{a_{1}}} {\sum^{n}_{i=0} e^{a_{i}}}" />
와 같이 표현 할 수 있다.


이 방법 외에 다른 방법으로도 유도가 가능하다.
이 방법은 우선  
<img src="https://latex.codecogs.com/svg.latex?\;\frac{y} {1-y} = e^t" />
의 확장이  
<img src="https://latex.codecogs.com/svg.latex?\;\frac{P(C_{i}\mid x)} {P(C_{k}\mid x)} = e^{t_{i}}" />
라는 것을 이해해야 한다.
각각 클래스가 2개일 때, 클래스가 K개 일 때 odds를 나타낸 것이다. 두 번쨰 식이 클래스가 K개 일 때 odds가 될 수 있는 이유는 K를 기준으로 그 외의 경우의 수들이 독립적인 이진 경우의 수라고 볼 수 있기 때문이다. 이것은 independence of irrelevant alternatives(IIA)라고 하는 가정에 기반하고 있다. 이 가정은 사회과학에 나오 것으로 간단하게 설명하면 A,B만 존재하는 상황과 A,B,C라는 세 가지 선택지가 존재는 상황에서 A와 B에 대한 선호는 동일하다는 것이다. 

<img src="https://latex.codecogs.com/svg.latex?\;\sum^{K-1}_{i=1} \frac{P(C_{i}\mid x)} {P(C_{k}\mid x)} = \sum^{K-1}_{i=1} e^{t_{i}}" />

좌면을 정리하면,

<img src="https://latex.codecogs.com/svg.latex?\;\sum^{K-1}_{i=1} \frac{P(C_{i}\mid x)} {P(C_{k}\mid x)} = \frac{\sum^{K-1}_{i=1} P(C_{i}\mid x)} {P(C_{k}\mid x)} = \frac{1-P(C_{K}\mid x)}{P(C_{K}\mid x)}" />
이 된다.

위 식을 다시 쓰면,

<img src="https://latex.codecogs.com/svg.latex?\;\frac{1-P(C_{K}\mid x)} {P(C_{K}\mid x)} = \sum^{K-1}_{i=1} e^{t_{i}}" />

이 되고, P(C_K | x) 기준으로 정리하면 다음과 같다.

<img src="https://latex.codecogs.com/svg.latex?\;P(C_{K}\mid x) = \frac{1} {1+\sum^{K-1}_{i=1}e^{t_{i}}}" />
<br>
<img src="https://latex.codecogs.com/svg.latex?\;P(C_{K}\mid x) = \frac {P(C_{i}\mid x)}{e^{t_{i}}}" />
이므로 

<img src="https://latex.codecogs.com/svg.latex?\;\frac{e^{t_{i}}} {1+\sum^{K-1}_{i=1}e^{t_{i}}} = \frac{e^{t_{i}}} {e^{t_{K}}+\sum^{K-1}_{i=1}e^{t_{i}}} = \frac{e^{t_{i}}} {\sum^{K}_{i=1}e^{t_{i}}}" />
가 되어 소프트 맥스 함수가 된다.

reference:
- 책
- https://de-novo.org/2018/05/03/logistic-cross-entropy-loss%EC%9D%98-%ED%99%95%EB%A5%A0%EB%A1%A0%EC%A0%81-%EC%9D%98%EB%AF%B8/
- https://opentutorials.org/module/3653/22995
- https://en.wikipedia.org/wiki/Multinomial_logistic_regression
