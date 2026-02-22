# digit-ml-drawer
<img width="1112" height="760" alt="image" src="https://github.com/user-attachments/assets/9ef74073-79fa-422b-b8d2-22484c986d1e" />

마우스로 쓴 숫자(0~9)를 실시간으로 인식하고, 각 숫자에 대한 유사율(확률) 막대 그래프를 보여주는 파이썬 데스크톱 앱입니다.

## 주요 기능

- `pygame` 캔버스에 손글씨 숫자 입력
- `0~9` 확률 분포 시각화(막대 그래프)
- 최고 확률 숫자와 신뢰도 표시
- 혼동이 잦은 `4/7/9`에 대한 형태 기반 후처리 보정
- 첫 실행 시 모델 학습 후 `digit_model.joblib` 캐시 저장

## 기술 스택

- Python 3.13+
- pygame
- scikit-learn (MLPClassifier)
- numpy, scipy, pillow, joblib

## 설치

```bash
cd /Users/oinse719/Desktop/digit-ml-drawer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python app.py
```

처음 실행 시 학습이 진행되어 수 초가 소요될 수 있습니다.

## 사용법

1. 검은 캔버스 중앙에 숫자를 크게 씁니다.
2. 마우스를 떼면 자동 예측됩니다. (`인식` 버튼으로 수동 예측 가능)
3. 오른쪽 그래프에서 `0~9` 확률 분포를 확인합니다.
4. `지우기` 버튼으로 초기화합니다.

## 정확도 팁

- 숫자는 캔버스의 `60~80%` 크기로 크게 쓰는 것이 좋습니다.
- `2`는 아랫꼬리, `4`는 가로획, `7`은 윗가로획을 분명하게 쓰면 인식이 안정적입니다.
- 계속 오인식되면 여러 번 써서 확률 분포가 어떻게 바뀌는지 비교해 보세요.

## 모델 재학습

모델 구조/전처리를 바꿨는데 이전 모델이 남아 있으면 캐시를 삭제하고 다시 실행하세요.

```bash
rm -f digit_model.joblib
python app.py
```

## 프로젝트 파일

- `app.py`: 학습, 전처리, 예측, UI 전체 로직
- `requirements.txt`: 의존성 목록
- `digit_model.joblib`: 학습된 모델 캐시(자동 생성)

## 라이선스

개인/학습용 예제 프로젝트입니다.
