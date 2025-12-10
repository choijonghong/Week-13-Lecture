# Week-12-Lecture
Week-12-Lecture

### 역전파는 어떻게 AI의 암흑기를 끝내고 딥러닝의 시작을 열었는가?
* 논문명: Learning representations by back-propagating errors”(Rumelhart, Hinton, Williams, 1986)

### 논문배경
* 당시 신경망은 입력과 출력만 연결한 구조에서만 학습이 가능했다.
    * ‘정답만 맞추는 연습’은 할 수 있었지만, ‘생각하는 과정’을 배우는 기능은 없었습니다.
* Hidden Layer가 추가되면 그 내부를 학습시키는 방법이 없었다.
    * 옛 신경망은 ‘정답’은 가르칠 수 있었지만, ‘생각 과정’을 어떻게 고쳐야 할지 알려줄 방법이 없었습니다.
* 기존 학습 규칙(예: 퍼셉트론)은 입력–출력 연결 강도만 조정할 수 있었다.
* 이로 인해 모델이 스스로 중간 개념을 배우는 표현 학습이 불가능했다.
* 본 논문은 Hidden Layer까지 학습 가능한 새로운 학습법 제안을 목표로 출발했다.

### 기존 연구의 한계점
* 당시 신경망은 ‘답만 외우는 학생’과 같아서, 생각하는 과정과 개념 만들기는 불가능했다.
* 중간에 어떤 생각 과정을 거쳤는지 확인할 수 없었다.
* 복잡한 문제를 단순 기준으로만 해결하려 했다.
* 스스로 새로운 아이디어나 특징을 만들지 못했다. 즉 사람이 알려주지 않으면 특징을 발견하지 못했다.

### 논문 목표
* Hidden Layer가 스스로 표현을 구축하도록 하는 학습 방법을 만드는 것이 목표다.
* 단순하면서도 다양한 다층 신경망에 적용 가능한 일반적 알고리즘을 제시하는 것이 목표다.

### 논문내용(주요 혁신 및 방법론)  :출력 오차를 역방향으로 전파하여 각 weight의 기여도를 계산하는 알고리즘
* Forward pass: 한 번 앞으로 계산해서 예측값을 만든다.
 <img width="768" height="190" alt="image" src="https://github.com/user-attachments/assets/4c1d47c1-21b8-4f39-b233-1acfac103f7f" />

* Backward pass: 오차를 뒤로 전달하며, 각 연결이 얼마나 잘못됐는지 미분으로 계산한다.
 <img width="830" height="250" alt="image" src="https://github.com/user-attachments/assets/a3c5a91a-b057-46c7-abd4-cf4be13cbd38" />

* Gradient descent 업데이트: 그 정보를 바탕으로 모든 weight를 오차가 줄어드는 방향으로 조금씩 고친다.
<img width="771" height="469" alt="image" src="https://github.com/user-attachments/assets/1015bcea-7347-4966-9f32-a89cfb6d0fc3" />

### 기여도 및 결과
* (역전파) Hidden Layer가 의미 있는 표현을 스스로 학습하도록 하는 방법을 제시했다.
* 다층 퍼셉트론 학습을 가능하게 하며 AI 발전의 전환점을 만들었다.
* 이 방식은 순환 신경망(RNN)에도 확장 가능함을 보여주었다.
* CNN, LSTM, Transformer 등 현대 딥러닝의 핵심 기반이 되는 학습 원리를 확립했다.
* AI가 스스로 특징을 학습하는 “Representation Learning” 시대의 출발점을 열었다.

### 감성컴퓨팅과 역전파
* 감성 신호(표정·음성·텍스트)에서 보이지 않는 감정 상태를 예측하기 위한 내부 표현을 스스로 학습하게 해준다.
* 감정 판단에 영향을 준 입력 요소가 무엇인지 오차를 역으로 계산해 weight를 조정한다.
* 복잡하고 모호한 감정 패턴을 다층 구조에서 점차 추상적 개념으로 학습하게 한다.
* 결국 AI가 ‘감정’이라는 숨겨진 개념을 데이터 기반으로 이해하고 일반화하도록 만든다.

### 참고 파일

* 논문1. Learning representations by back-propagating errors
* 논문2. Improving Students’ Daily Life Stress Forecasting using  LSTM  Neural  Networks
     * 스트레스는 인간 감정 상태의 핵심 신호이며, 이를 예측하는 것은 감성컴퓨팅의 중요한 목표이다.
     * LSTM은 일상 데이터 흐름 속 변화 패턴을 학습하여 **미래 감정 상태(스트레스)**를 예측할 수 있음을 보였다.
     * 특히 설문 없이 웨어러블·모바일의 객관적 신호만으로 감정 추정을 가능하게 했다.
     * 이는 감정 이해를 ‘현재 상태 측정’에서 **‘미래 감정 전망’**으로 확장하는 연구다.
     * 결과적으로, 감성컴퓨팅을 개인의 행동 개입·예방적 웰빙 지원으로 연결하는 실질적 기여를 했다. 
