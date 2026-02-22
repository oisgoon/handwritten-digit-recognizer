# digit-ml-drawer

마우스로 손글씨 숫자(0~9)를 쓰면, 모델이 어떤 숫자인지 예측하고 0~9 유사율(확률) 그래프를 보여주는 데모 앱입니다.

## 1) 설치

```bash
cd /Users/oinse719/Desktop/digit-ml-drawer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 실행

```bash
python app.py
```

처음 실행 시 `scikit-learn digits` 데이터셋으로 간단한 MLP 모델을 학습하고 `digit_model.joblib`로 저장합니다.

## 3) 사용법

1. 검은 캔버스 중앙에 숫자를 크게 씁니다.
2. 마우스를 떼면 자동 예측(또는 `인식` 버튼).
3. 오른쪽 그래프에서 0~9 확률 분포를 확인합니다.
4. `지우기`로 다시 시도합니다.
