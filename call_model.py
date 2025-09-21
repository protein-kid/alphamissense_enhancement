import joblib
import numpy as np

# 전처리 함수
def tau(v):
    if v.sum() <= 0:
        return 0.0
    p = v / (v.sum() + 1e-12)
    H = -(p*np.log(p+1e-12)).sum()
    return 1.0 - H/np.log(len(v))

# 모델 불러오기
model = joblib.load("rf_model.pkl")
tissue_cols = joblib.load("tissue_columns.pkl")  # 학습에 사용한 조직 리스트

# 사용자 입력 받기
am_score = float(input("AlphaMissense 점수를 입력하세요 (예: 0.812): "))

print("\n조직 발현 정도를 입력하세요 (0,1,2,3 중 하나). 입력하지 않으면 0으로 처리됩니다.")
tissue_values = []
for col in tissue_cols:
    val = input(f"{col}: ")
    if val.strip() == "":
        val = 0
    tissue_values.append(int(val))

# Feature 구성
T = np.array(tissue_values, dtype=float).reshape(1, -1) / 3.0
taus = np.array([tau(T.flatten())]).reshape(-1,1)
sparsity = np.array([(T <= 0).mean()]).reshape(-1,1)

X = np.hstack([np.array([[am_score]]), T, taus, sparsity])

# 예측
pred_prob = model.predict_proba(X)[:,1][0]

print("\n=== 예측 결과 ===")
print(f"병리성 가능성 (0~1): {pred_prob:.4f}")
if pred_prob > 0.5:
    print("병리성 가능성이 높습니다.")
else:
    print("양성 가능성이 높습니다.")
