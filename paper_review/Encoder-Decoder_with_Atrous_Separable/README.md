### 어떠한 문제를 해결하고자 했는가?

segmentation task에 두 가지 방법이 있다. SPP와 encoder-decoder이다.
SPP는 sharper object boundary를 찾을 수 있고, encdoer-decoder는 더 많은 context 정보를 추출할 수 있다. 기존 deeplabv3는 SPP였을 것으로 추정되는데(시간이 없어서 deeplabv3를 아직 못봄..decoder output stride 16 이었나봄. 이것이 성공적으로 디테일을 복원하지 못했다) 이것을 encoder로 사용하고 decoder를 추가적으로 사용하여 object boundary를 더 잘 찾고자 함

### 그 문제를 해결한 방법은 무엇인가?

기존 deeplabv3는 SPP였을 것으로 추정되는데(시간이 없어서 deeplabv3를 아직 못봄..) 이것을 encoder로 사용하고 decoder를 추가적으로 사용하여 object boundary를 더 잘 찾고자 함
Resnet-101대신 XceptionNet 사용.
Separable 개념도 +에서 처음 나옴.


### 그 방법에 대한 이해 without 수학 (intuition)

### 그 방법에 대한 이해 with 수학

### 제안한 방법이 어떠한 결과를 내었는가? (특히 어떤 부분이 좋아졌는가..?)

### 부족한 부분에는 무엇이 있나?

#### encoder-decoder 나온 이유
fc 통과하면 2차원 배열 구조가 사라져서 이미지 공간 정보 소멸해 부적합. fc는 멀리 있는 픽셀과도 연결되서 연산되기 떄문. DCN은 위치에 따라서 stack되기 때문. 그냥 fc, pooling 없애는 것으로는 연산량이 너무 많다. fully convolution network에 의해서 이미지 크기와 위치에 상관이 없이지게 됨. FCN문제점 유리에 비친 물체를 개별적인 물체로 인식 및 매우 작은 물체 인식 불가.
