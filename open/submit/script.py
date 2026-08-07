"""
[제출] LightGBM + XGBoost 블렌드 — 추론 스크립트

평가 서버가 실행하는 코드. ./data/test.csv 를 읽어 ./output/submission.csv 를 만든다.
전처리(콜드스타트 플래그, count/leverage 교차 피처, median 대치, OrdinalEncoder)는
train_final.py 가 만든 ./model/artifacts.pkl 안의 통계/객체를 그대로 재사용한다 (전부 train 기준 fit).
"""
import os

# macOS(Apple Silicon) 로컬 테스트 환경에서 LightGBM/XGBoost가 번들한 libomp와
# 기존 OpenMP 런타임이 충돌해 joblib.load / predict 시점에 데드락이 나는 걸 확인했다
# (스레드 4개 이상에서 100% 재현). 평가 서버(Ubuntu, 별도 컨테이너)에서는 발생 안 할 가능성이
# 높지만, 무해한 방어 코드라 그대로 둔다 — 반드시 각 라이브러리 import 전에 설정해야 한다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


def build_features(df, art):
    df = df.copy()

    na_cols = [c for c in art["na_cols"] if c in df.columns]
    pitcher_na = art["pitcher_na"]
    batter_na = art["batter_na"]
    df["pitcher_is_cold_start"] = df[pitcher_na].isna().any(axis=1).astype(int)
    df["batter_is_cold_start"] = df[batter_na].isna().any(axis=1).astype(int)

    df[na_cols] = df[na_cols].fillna(art["medians"])

    df["count_state"] = df["balls_before"] * 10 + df["strikes_before"]
    df["base_x_outs"] = df["base_state"].astype(str) + "_" + df["outs_before"].astype(str)
    df["is_high_leverage"] = (df["li"] >= art["li_q75"]).astype(int)
    df["is_close_late"] = ((df["inning"] >= 7) & (df["score_diff_pitcher_team"].abs() <= 1)).astype(int)
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    # 팀 asof 성공률: test(2025)는 train(2019~2024) 전체보다 뒤이므로, 각 팀의
    # train 전체 누적 성공률(고정값)을 그대로 asof 값으로 사용한다.
    # (참고: 선수 단위 손잡이 매치업 asof는 v3에서 로컬-실전 괴리로 폐기 — SUBMISSIONS.md 참고)
    med = art["group_asof_medians"]
    df["asof_team_pitcher_success_rate"] = df["pitcher_team_id"].map(art["team_pitcher_rate_lookup"]).fillna(med["asof_team_pitcher_success_rate"])
    df["asof_team_batter_success_rate"] = df["batter_team_id"].map(art["team_batter_rate_lookup"]).fillna(med["asof_team_batter_success_rate"])

    cat_cols = art["cat_cols"]
    df[cat_cols] = art["encoder"].transform(df[cat_cols])

    return df[art["feats"]]


def predict(X, art):
    p_lgb = art["lgb_model"].predict_proba(X)[:, 1]
    p_xgb = art["xgb_model"].predict_proba(X)[:, 1]
    # CatBoost는 cat_features가 int/str만 허용 (ordinal encoder 출력은 float)
    X_cb = X.copy()
    X_cb[art["cat_cols"]] = X_cb[art["cat_cols"]].astype(int)
    p_cb = art["cb_model"].predict_proba(X_cb)[:, 1]
    # v7: LGB+XGB+CatBoost 3개를 K-fold OOF로 학습한 로지스틱 메타러너로 결합
    pred = art["meta_model"].predict_proba(np.column_stack([p_lgb, p_xgb, p_cb]))[:, 1]

    # bias-fix: 학습 시즌 추세로 추정한 기대 성공률로 예측 평균을 맞춤 (train 통계만 사용)
    shift = art["expected_rate_2025"] - pred.mean()
    pred = np.clip(pred + shift, 1e-4, 1 - 1e-4)
    return pred


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    ARTIFACT_PATH = os.path.join(MODEL_DIR, "artifacts.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load artifacts...")
    art = joblib.load(ARTIFACT_PATH)
    print(f" OK. features={len(art['feats'])}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, art)
    print(f" features={X.shape[1]}")

    print("Inference model...")
    preds = predict(X, art) if len(X) else []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
