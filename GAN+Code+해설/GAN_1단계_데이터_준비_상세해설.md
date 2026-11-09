# GAN 1단계: MNIST 데이터 준비 코드 상세 해설

## 1. 학습 목표

GAN 학습의 첫 번째 단계는 **MNIST 손글씨 숫자 데이터를 불러오고 신경망 학습에 적합한 형태로 변환하는 것**이다. 이 단계에서는 아직 생성자와 판별자의 가중치를 학습하지 않는다. 판별자가 실제 손글씨 이미지를 학습에 활용할 수 있도록 데이터를 준비한다.

## 2. 전체 코드

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 하이퍼파라미터 설정
epochs = 100
lr = 0.0002
batch_size = 128
latent_dim = 100

# 2. 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. MNIST 데이터셋 불러오기
train_dataset = datasets.MNIST(
    root='./mnist_data/',
    train=True,
    transform=transform,
    download=True
)

# 4. 데이터 로더 생성
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
```

## 3. 하이퍼파라미터 설정

| 코드 | 설정값 | 의미 |
|---|---:|---|
| `epochs` | 100 | 전체 학습 데이터를 100번 반복 |
| `lr` | 0.0002 | 가중치 수정에 사용하는 학습률 |
| `batch_size` | 128 | 한 번에 처리하는 이미지 수 |
| `latent_dim` | 100 | 생성자에 입력하는 무작위 벡터의 차원 |

**예시:** 학생 60,000명의 손글씨 자료를 한 번에 모두 처리하는 대신 128장씩 나누어 학습한다. 전체 자료를 한 바퀴 학습하면 1에포크이며, 이를 100번 반복한다.

`latent_dim = 100`은 이미지의 픽셀 개수가 아니라, 생성자가 숫자 이미지를 만들 때 사용하는 잠재벡터의 차원이다.

## 4. 데이터 전처리

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### 4.1 `transforms.Compose()`

여러 데이터 변환 작업을 정해진 순서대로 실행한다. 이 코드에서는 (1) 텐서 변환, (2) 정규화를 수행한다.

### 4.2 `transforms.ToTensor()`

MNIST 이미지를 PyTorch에서 계산할 수 있는 텐서로 변환한다. MNIST 이미지는 **28 × 28 크기의 흑백 이미지**다. 원래 픽셀값은 0~255이며, `ToTensor()`는 이를 0~1 사이의 실수로 변환한다.

| 원래 픽셀값 | 텐서 변환 후 |
|---:|---:|
| 0 | 0.0 |
| 128 | 약 0.502 |
| 255 | 1.0 |

한 장의 이미지 텐서 형태는 `(1, 28, 28)`이다. 첫 번째 1은 흑백 채널 수를 의미한다.

### 4.3 `transforms.Normalize((0.5,), (0.5,))`

텐서의 픽셀값을 다시 -1부터 1 사이로 변환한다.

$$
x_{\text{정규화}} = \frac{x-0.5}{0.5}
$$

| 원래 픽셀값 | `ToTensor()` 후 | 정규화 후 |
|---:|---:|---:|
| 0 | 0 | -1 |
| 128 | 약 0.502 | 약 0.004 |
| 255 | 1 | 1 |

**왜 -1부터 1로 정규화할까?** 제공된 생성자의 마지막 활성화 함수가 `nn.Tanh()`이기 때문이다. Tanh 출력 범위는 -1부터 1까지이므로, 실제 이미지와 생성 이미지의 값 범위를 맞춘다.

## 5. MNIST 데이터셋 불러오기

```python
train_dataset = datasets.MNIST(
    root='./mnist_data/',
    train=True,
    transform=transform,
    download=True
)
```

| 코드 | 의미 |
|---|---|
| `datasets.MNIST` | MNIST 손글씨 데이터셋 사용 |
| `root` | 데이터를 저장할 폴더 |
| `train=True` | 학습용 데이터 60,000장 사용 |
| `transform=transform` | 앞서 정의한 전처리 적용 |
| `download=True` | 데이터가 없으면 다운로드 |

MNIST에는 숫자 0~9까지 여러 사람의 다양한 필기 형태가 포함되어 있다. 예를 들어 같은 숫자 3이라도 곡선, 기울기, 굵기가 다르다. GAN은 이 이미지들의 시각적 특징과 데이터 분포를 학습하도록 설계된다.

**중요:** 이 코드에서는 숫자 정답 레이블을 GAN의 학습 목표로 사용하지 않는다. 학습 루프의 `for i, (imgs, _) in enumerate(train_loader):`에서 `_`는 레이블을 사용하지 않음을 나타낸다.

## 6. DataLoader: 데이터를 128장씩 나누기

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
```

- `dataset=train_dataset`: MNIST 학습 데이터 사용
- `batch_size=128`: 한 번에 128장씩 공급
- `shuffle=True`: 매 에포크마다 데이터 순서를 무작위로 섞음

60,000 ÷ 128 = 468.75이므로 **한 에포크에는 469개 배치**가 있다. 앞의 468개 배치에는 각 128장이, 마지막 배치에는 96장이 들어간다. 100에포크에서는 총 **46,900번의 배치 처리**가 수행된다.

`shuffle=True`는 새로운 숫자 이미지를 생성하는 기능이 아니라 기존 이미지의 순서를 섞는 기능이다.

## 7. 데이터 준비와 GAN 학습의 관계

```text
MNIST 실제 손글씨 60,000장
          ↓
텐서 변환 및 정규화
(28×28 픽셀, -1~1 범위)
          ↓
128장씩 배치 구성
          ↓
실제 이미지 ─────────────┐
                        ├→ 판별자: 진짜·가짜 구분 학습
생성자가 만든 가짜 이미지 ┘
```

데이터 준비 단계에서는 **실제 이미지만 준비**한다. 가짜 이미지는 이후 생성자가 무작위 잠재벡터를 입력받아 만든다.

## 8. 실습 확인 코드

```python
# 첫 번째 배치 확인
imgs, labels = next(iter(train_loader))
print('이미지 배치 형태:', imgs.shape)   # 일반적으로 [128, 1, 28, 28]
print('레이블 배치 형태:', labels.shape) # 일반적으로 [128]
print('픽셀 최솟값:', imgs.min().item())
print('픽셀 최댓값:', imgs.max().item())

# 첫 번째 숫자 이미지 확인
import matplotlib.pyplot as plt
plt.imshow(imgs[0].squeeze(), cmap='gray', vmin=-1, vmax=1)
plt.title(f'MNIST label: {labels[0].item()}')
plt.axis('off')
plt.show()
```

레이블은 데이터 확인을 위해 출력하지만, 제공된 GAN 학습 코드에서는 판별자 학습의 숫자 종류 정답으로 사용하지 않는다.

## 9. 핵심 요약 5문장

1. MNIST에서 숫자 0~9의 손글씨 이미지 60,000장을 불러온다.
2. 이미지를 PyTorch 텐서로 변환하여 신경망이 계산할 수 있게 한다.
3. 픽셀값을 -1부터 1 사이로 정규화한다.
4. 학습 데이터를 무작위로 섞고 128장씩 나누어 공급한다.
5. **준비한 실제 이미지는 이후 판별자가 진짜와 가짜를 구분하도록 학습하는 데 사용된다.**
