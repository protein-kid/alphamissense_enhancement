import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

# 데이터 로딩
clinvar = pd.read_csv("alphamissense_eval/clinvar_with_am.tsv", sep="\t")
tissue  = pd.read_csv("clinvar_with_am_tissue_matrix.tsv", sep="\t")
df = clinvar.merge(tissue, on="protein_id", how="left")

# Feature 구성
tissue_cols = [c for c in tissue.columns if c not in ["protein_id","ensg"]]

# 조건: 모든 조직 값이 NaN이거나 전부 0인 경우 제외
mask_has_tissue = df[tissue_cols].notna().any(axis=1) & (df[tissue_cols].sum(axis=1) > 0)
df = df.loc[mask_has_tissue].copy()

print(f"조직 정보가 있는 변이 수: {len(df)} / 전체 {len(clinvar)}")

# 조직 데이터 전처리
T = df[tissue_cols].to_numpy(dtype=float)
T = np.nan_to_num(T, nan=0.0)
Tn = T / 3.0  # 값 정규화 (0~1)

# 조직 다양성 지표 (tau)
def tau(v):
    if v.sum() <= 0: return 0.0
    p = v / (v.sum() + 1e-12)
    H = -(p*np.log(p+1e-12)).sum()
    return 1.0 - H/np.log(len(v))

taus = np.apply_along_axis(tau, 1, T)
sparsity = (T <= 0).mean(axis=1)  # 비발현 비율

# 입력 feature
X_am     = df["alphamissense"].values.reshape(-1,1)                     # A: AM 단독
X_tissue = np.hstack([Tn, taus.reshape(-1,1), sparsity.reshape(-1,1)])  # B: Tissue 단독
X_both   = np.hstack([X_am, X_tissue])                                  # C: AM + Tissue

y = df["label"].values
groups = df["protein_id"].values

# 교차검증으로 성능 비교
def eval_model(X, y, groups, name):
    gkf = GroupKFold(n_splits=5)
    aucs, prs = [], []
    for tr, te in gkf.split(X, y, groups):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X[tr], y[tr])
        preds = model.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], preds))
        prs.append(average_precision_score(y[te], preds))
    return name, np.mean(aucs), np.std(aucs), np.mean(prs), np.std(prs)

results = []
results.append(eval_model(X_am, y, groups, "A: AlphaMissense only"))
results.append(eval_model(X_tissue, y, groups, "B: Tissue only"))
results.append(eval_model(X_both, y, groups, "C: AM + Tissue"))

res_df = pd.DataFrame(results, columns=["Model","ROC-AUC mean","ROC-AUC std","PR-AUC mean","PR-AUC std"])
print("\n=== 성능 비교 ===")
print(res_df)

# 최종 모델 학습 및 저장
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_both, y)  # 보완 모델 전체 데이터로 학습

# 모델 저장
joblib.dump(final_model, "rf_model.pkl")
print("\n최종 보완 모델이 rf_model.pkl 파일로 저장되었습니다.")

# feature 컬럼 리스트도 함께 저장하는 것을 추천
joblib.dump(tissue_cols, "tissue_columns.pkl")
print("사용한 feature 컬럼 리스트도 tissue_columns.pkl 로 저장되었습니다.")
