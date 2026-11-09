# 4단계. GAN 경쟁 학습(Adversarial Training) 코드 상세 해설

## 1. 경쟁 학습이란?

**GAN의 경쟁 학습은 생성자(Generator, G)와 판별자(Discriminator, D)가 서로 다른 목표로 번갈아 학습하면서, 생성자가 실제 MNIST 손글씨와 유사한 숫자 이미지를 만들도록 하는 과정**이다.

- **생성자 G:** 가짜 숫자를 진짜처럼 만들어 판별자를 속이려고 학습한다.
- **판별자 D:** 실제 숫자와 생성자가 만든 숫자를 구분하도록 학습한다.

판별자가 생성자에게 언어로 수정 지시를 주는 것이 아니다. **판별자를 거쳐 계산한 손실함수의 기울기가 역전파되어 생성자 가중치의 수정 방향을 결정**한다.

## 2. 전체 학습 흐름

1. MNIST에서 실제 이미지 배치를 가져온다.
2. 무작위 잠재벡터를 생성자에 넣어 가짜 이미지를 만든다.
3. 판별자가 실제 이미지는 1, 가짜 이미지는 0으로 구분하도록 학습한다.
4. 생성자가 만든 가짜 이미지를 판별자에 다시 넣는다.
5. 이번에는 생성자가 판별자 출력이 1에 가까워지도록 학습한다.
6. 배치마다 이 과정을 반복한다.

## 3. 경쟁 학습 코드의 핵심 구조

다음 코드는 앞서 설명한 GAN 실습의 학습 구조를 이해하기 쉽도록 정리한 것이다.

```python
criterion = nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

for epoch in range(epochs):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size_now = real_imgs.size(0)

        real_labels = torch.ones(batch_size_now, 1, device=device)
        fake_labels = torch.zeros(batch_size_now, 1, device=device)

        # 1. 판별자 학습
        optimizer_D.zero_grad()
        real_output = discriminator(real_imgs)
        d_real_loss = criterion(real_output, real_labels)

        z = torch.randn(batch_size_now, latent_dim, device=device)
        fake_imgs = generator(z)
        fake_output = discriminator(fake_imgs.detach())
        d_fake_loss = criterion(fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # 2. 생성자 학습
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_imgs)
        g_loss = criterion(fake_output, real_labels)

        g_loss.backward()
        optimizer_G.step()
```

이 구조에서는 **한 배치마다 판별자 1회, 생성자 1회** 최적화한다. 생성자 학습 시 판별자를 통과해 기울기를 구하지만 `optimizer_G.step()`은 생성자 매개변수만 갱신한다.

## 4. 판별자 학습: 실제 이미지와 가짜 이미지

### 4.1 실제 이미지 학습

```python
real_output = discriminator(real_imgs)
d_real_loss = criterion(real_output, real_labels)
```

실제 MNIST 이미지의 정답은 **1**이다. 판별자 출력이 0.9이면 정답에 가까워 손실이 작고, 0.1이면 손실이 크다.

### 4.2 가짜 이미지 학습

```python
z = torch.randn(batch_size_now, latent_dim, device=device)
fake_imgs = generator(z)
fake_output = discriminator(fake_imgs.detach())
d_fake_loss = criterion(fake_output, fake_labels)
```

- `torch.randn(...)`: 100차원 무작위 잠재벡터를 배치 수만큼 생성한다.
- `generator(z)`: 잠재벡터를 가짜 숫자 이미지로 바꾼다.
- `fake_imgs.detach()`: 판별자 학습 중 생성자까지 기울기가 전파되는 것을 막는다.
- `fake_labels`: 가짜 이미지의 정답 **0**이다.

### 4.3 판별자 가중치 갱신

```python
d_loss = d_real_loss + d_fake_loss
d_loss.backward()
optimizer_D.step()
```

판별자는 실제 이미지에 대한 손실과 가짜 이미지에 대한 손실을 더한 뒤 역전파하고, Adam으로 판별자 가중치를 수정한다.

$$
L_D=L_{\mathrm{real}}+L_{\mathrm{fake}}
$$

## 5. 생성자 학습: 가짜 이미지를 진짜처럼 만들기

```python
optimizer_G.zero_grad()
fake_output = discriminator(fake_imgs)
g_loss = criterion(fake_output, real_labels)
g_loss.backward()
optimizer_G.step()
```

생성자는 가짜 이미지를 판별자에 넣지만, 목표 레이블은 **0이 아니라 1**로 설정한다. 생성자는 판별자가 자신의 이미지를 실제 이미지처럼 판단하도록 학습하기 때문이다.

예를 들어 가짜 이미지에 대한 판별자 출력이 0.1이라면 생성자 손실은 크고, 0.9라면 작다. 생성자 학습 시 판별자의 가중치는 이 코드의 `optimizer_G.step()`으로 갱신되지 않는다.

## 6. 손실함수: BCELoss

이진 교차 엔트로피(Binary Cross-Entropy)의 한 사례에 대한 식은 다음과 같다.

$$
L=-[y\ln(p)+(1-y)\ln(1-p)]
$$

- $y$: 정답 레이블(0 또는 1)
- $p$: 판별자 출력값
- $\ln$: 자연로그

### 6.1 판별자 손실

$$
L_D=-\ln(D(x))-\ln(1-D(G(z)))
$$

- $x$: 실제 이미지
- $z$: 무작위 잠재벡터
- $G(z)$: 생성자가 만든 이미지
- $D(x)$: 실제 이미지에 대한 판별값
- $D(G(z))$: 가짜 이미지에 대한 판별값

판별자는 $D(x)$를 1에 가깝게, $D(G(z))$를 0에 가깝게 만들려고 한다. 위 식은 실제·가짜 이미지가 각각 한 개일 때의 합산 예시이며, 배치에서는 각 항에 대한 평균 손실을 사용한다.

### 6.2 생성자 손실

$$
L_G=-\ln(D(G(z)))
$$

생성자는 가짜 이미지에 대한 판별값을 1에 가깝게 만들려고 한다. 이는 **비포화(non-saturating) GAN 생성자 손실**이다.

## 7. 숫자로 이해하는 경쟁 학습

가짜 숫자 이미지 하나에 대한 판별자 출력 $p=D(G(z))$를 가정한다.

- 판별자의 가짜 이미지 손실: $-\ln(1-p)$
- 생성자 손실: $-\ln(p)$

| 가짜 이미지 판별값 $p$ | 판별자 손실 $-\ln(1-p)$ | 생성자 손실 $-\ln(p)$ |
|---:|---:|---:|
| 0.1 | 0.105 | 2.303 |
| 0.3 | 0.357 | 1.204 |
| 0.5 | 0.693 | 0.693 |
| 0.7 | 1.204 | 0.357 |
| 0.9 | 2.303 | 0.105 |

**해석:** 같은 판별값이라도 판별자는 가짜 이미지에 대해 0에 가까운 출력을 원하고, 생성자는 1에 가까운 출력을 원한다. 이 표의 판별자 손실에는 실제 이미지 손실이 포함되지 않았다.

## 8. 학습 명령어 3개의 의미

| 명령어 | 기능 | 쉬운 의미 |
|---|---|---|
| `zero_grad()` | 기존 기울기 초기화 | 이전 계산에서 누적된 기울기 지우기 |
| `backward()` | 역전파로 기울기 계산 | 가중치 수정 방향과 변화율 계산 |
| `step()` | Adam으로 가중치 수정 | 계산한 기울기를 이용해 실제 가중치 변경 |

예를 들어 판별자가 가짜 이미지를 진짜라고 잘못 판단하면 판별자 손실이 커진다. 역전파는 해당 손실을 줄이기 위한 가중치 기울기를 계산하고, Adam이 가중치를 수정한다.

## 9. 100에포크 학습의 의미

- 학습 이미지: **60,000장**
- 배치 크기: **128장**
- 한 에포크의 배치 수: **469개** (마지막 배치 96장)
- 학습 횟수: **100에포크**
- 총 배치 처리 횟수: **46,900회**

각 배치에서 판별자와 생성자 최적화가 각각 한 번씩 이루어진다. 학습이 진행되면서 생성 이미지가 더 그럴듯해질 수 있지만, GAN은 학습이 불안정하거나 다양성이 줄어드는 현상(모드 붕괴)이 나타날 수도 있으므로 개선이 보장되지는 않는다.

## 10. 경쟁 학습과 숫자 생성·변환의 관계

**경쟁 학습은 생성 능력을 얻는 방법이고, 숫자 생성은 학습 결과이며, 숫자 변환은 학습된 생성자를 활용하는 실험**이다.

1. 경쟁 학습으로 생성자의 가중치가 수정된다.
2. 학습 후 생성자에 새로운 무작위 잠재벡터를 넣으면 숫자 이미지를 만들 수 있다.
3. 서로 다른 잠재벡터 두 개를 보간하면 이미지가 점진적으로 변화하는 과정을 관찰할 수 있다.
4. 숫자 생성이나 보간 단계에서는 판별자를 다시 경쟁 학습시킬 필요가 없다.
5. 이 코드에서는 숫자 3이나 8 같은 특정 레이블을 직접 지정하지 않는다.

## 11. 요약하면,

1. GAN 경쟁 학습은 생성자와 판별자가 서로 다른 목표로 번갈아 학습하는 과정이다.
2. 생성자는 가짜 숫자를 실제 손글씨처럼 만들려고 한다.
3. 판별자는 실제 숫자와 생성된 숫자를 구분하려고 한다.
4. 판별자는 실제 이미지에 정답 1, 가짜 이미지에 정답 0을 사용한다.
5. 생성자는 자신의 가짜 이미지에 대해 판별자가 1을 출력하도록 학습한다.
6. 두 신경망은 BCELoss로 각각의 손실을 계산한다.
7. 역전파는 손실을 줄이기 위한 가중치 기울기를 계산한다.
8. Adam은 계산된 기울기를 이용해 각 신경망의 가중치를 수정한다.
9. 이 과정을 100에포크 반복하면서 생성자의 숫자 생성 능력이 개선될 수 있다.
10. **GAN의 핵심은 판별자의 평가를 학습 신호로 활용하여 생성자가 실제 데이터와 유사한 이미지를 생성하도록 학습한다는 점이다.**

---

**결론:** GAN의 경쟁은 단순히 승패를 결정하는 과정이 아니라 생성자가 새로운 이미지를 만들어내는 능력을 학습하기 위한 최적화 과정이다.
