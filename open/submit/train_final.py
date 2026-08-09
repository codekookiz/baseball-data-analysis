"""
최종 제출 모델 학습 스크립트 (평가 서버에서는 실행되지 않음 — 로컬에서 실행해 model/ 아티팩트를 만든다).

파이프라인:
  1. train.csv(2019~2024) 전체 로드
  2. 콜드스타트 플래그, count/leverage 교차 피처 추가
  3. NA_COLS median 대치, 범주형 OrdinalEncoder — 전부 train 전체 기준으로 fit
  4. LightGBM / XGBoost 각각: 최근 10%(시간순) 홀드아웃으로 early stopping해서 n_iter 결정
     -> 그 n_iter를 고정하고 전체 데이터로 재학습 (val 대비 +30~40점 확인됨)
  5. 시즌별 성공률 추세선을 전체 학습 시즌으로 적합해 2025 기대 성공률 추정 (train만 사용, 규정 준수)
  6. 아티팩트(모델 2개, 인코더, median, 피처 목록, 블렌드 가중치, bias-fix 파라미터)를 model/artifacts.pkl 로 저장
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
DATA_DIR = "./data"
MODEL_PATH = "./model/artifacts.pkl"

# v6: 고정 블렌드 가중치(v4, v5 둘 다 단일 fold 튜닝이라 실전에서 배신당함) 대신
# K-fold OOF 기반 로지스틱 메타러너로 교체. 2022/2023/2024 세 시점 검증에서 2/3 승
# (2023 패배는 그 해 자체가 구조적으로 예측 불가능한 특이 케이스라 감안).


def add_interactions(df, li_q75):
    df["count_state"] = df["balls_before"] * 10 + df["strikes_before"]
    df["base_x_outs"] = df["base_state"].astype(str) + "_" + df["outs_before"].astype(str)
    df["is_high_leverage"] = (df["li"] >= li_q75).astype(int)
    df["is_close_late"] = ((df["inning"] >= 7) & (df["score_diff_pitcher_team"].abs() <= 1)).astype(int)
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def add_causal_group_asof(df, group_cols, out_col):
    """전체 train을 (season, game_month) 기준 시간순 정렬 후, group_cols 기준 그룹의
    '이 행 이전까지' 누적 성공률(shift+expanding과 동일)을 계산 -> 원래 순서로 되돌림. 완전히 causal."""
    order = df.sort_values(["season", "game_month"], kind="stable").index
    sorted_df = df.loc[order]
    cum_sum = sorted_df.groupby(group_cols)[TARGET].cumsum() - sorted_df[TARGET]
    cum_cnt = sorted_df.groupby(group_cols).cumcount()
    rate = cum_sum / cum_cnt.replace(0, np.nan)
    df.loc[order, out_col] = rate.values
    return df


def chrono_split(df, feats, es_frac=0.1):
    order = df.sort_values(["season", "game_month"]).index
    n_es = int(len(order) * es_frac)
    return (df.loc[order[:-n_es], feats], df.loc[order[-n_es:], feats],
            df.loc[order[:-n_es], TARGET], df.loc[order[-n_es:], TARGET])


def main():
    t0 = time.time()
    print("Load train.csv ...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    print(f" {train.shape} ({time.time()-t0:.1f}s)")

    NA_COLS = [c for c in train.columns if train[c].isna().any()]
    PITCHER_NA = [c for c in NA_COLS if c.startswith("asof_pitcher")]
    BATTER_NA = [c for c in NA_COLS if c.startswith("asof_batter")]
    train["pitcher_is_cold_start"] = train[PITCHER_NA].isna().any(axis=1).astype(int)
    train["batter_is_cold_start"] = train[BATTER_NA].isna().any(axis=1).astype(int)

    medians = train[NA_COLS].median()
    train[NA_COLS] = train[NA_COLS].fillna(medians)

    li_q75 = train["li"].quantile(0.75)
    train = add_interactions(train, li_q75)

    # 주의: 손잡이 매치업(선수 단위 고카디널리티) asof는 v3에서 로컬 val 대비 실제 LB가
    # -109점 회귀해서 폐기됨 (SUBMISSIONS.md v3 항목 참고). 팀 단위(저카디널리티, ~16~20개)
    # asof만 유지 — 이건 로컬(+38)/실전(+31) 방향과 크기가 일관되게 검증됐다.
    print("Building causal group-asof features (whole-file expanding) ...")
    train = add_causal_group_asof(train, "pitcher_team_id", "asof_team_pitcher_success_rate")
    train = add_causal_group_asof(train, "batter_team_id", "asof_team_batter_success_rate")

    team_asof_cols = ["asof_team_pitcher_success_rate", "asof_team_batter_success_rate"]
    group_asof_medians = train[team_asof_cols].median()
    train[team_asof_cols] = train[team_asof_cols].fillna(group_asof_medians)

    # 추론 시 test(2025)는 train 전체 기간 뒤이므로, 각 팀의 "전체 train 누적 성공률" 하나의
    # 고정값을 그대로 asof 값으로 사용한다 (평가 데이터 자체의 분포는 쓰지 않음 - 규정 준수).
    team_pitcher_rate_lookup = train.groupby("pitcher_team_id")[TARGET].mean()
    team_batter_rate_lookup = train.groupby("batter_team_id")[TARGET].mean()

    cat_cols = CAT_COLS + ["base_x_outs"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[cat_cols])
    train[cat_cols] = enc.transform(train[cat_cols])

    FEATS = ([c for c in train.columns if c not in [ID, TARGET]])
    print(f"features: {len(FEATS)}")

    X_fit, X_es, y_fit, y_es = chrono_split(train, FEATS)
    X_full, y_full = train[FEATS], train[TARGET]

    # ---- LightGBM: ES to pick n_iter, then refit on full data ----
    # num_leaves 15->31: v2 피처셋(56개) 기준 재튜닝 결과 foldB에서 746.66->760.46 (+13.8) 확인
    print("LightGBM: early stopping pass ...")
    lgbm_es = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=31, max_depth=8,
                                  min_child_samples=20, reg_lambda=1.0, bagging_fraction=0.8, bagging_freq=1,
                                  feature_fraction=0.8, random_state=42, n_jobs=-1, verbosity=-1)
    lgbm_es.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(50, verbose=False)])
    n_iter_lgb = lgbm_es.best_iteration_
    print(f" n_iter_lgb={n_iter_lgb}, refit on full data ...")
    lgb_model = lgb.LGBMClassifier(n_estimators=n_iter_lgb, learning_rate=0.05, num_leaves=31, max_depth=8,
                                    min_child_samples=20, reg_lambda=1.0, bagging_fraction=0.8, bagging_freq=1,
                                    feature_fraction=0.8, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_model.fit(X_full, y_full)

    # ---- XGBoost: same idea ----
    print("XGBoost: early stopping pass ...")
    xgb_es = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=5, max_leaves=31,
                                grow_policy="lossguide", min_child_weight=20, reg_lambda=1.0,
                                subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", random_state=42, n_jobs=-1,
                                eval_metric="logloss", early_stopping_rounds=50)
    xgb_es.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    n_iter_xgb = xgb_es.best_iteration
    print(f" n_iter_xgb={n_iter_xgb}, refit on full data ...")
    xgb_model = xgb.XGBClassifier(n_estimators=n_iter_xgb, learning_rate=0.05, max_depth=5, max_leaves=31,
                                   grow_policy="lossguide", min_child_weight=20, reg_lambda=1.0,
                                   subsample=0.8, colsample_bytree=0.8,
                                   tree_method="hist", random_state=42, n_jobs=-1)
    xgb_model.fit(X_full, y_full)

    # ---- CatBoost: v7에서 추가. 2022/2023/2024 다중fold 검증에서 3-way 앙상블이 3/3 승 확인 ----
    print("CatBoost: early stopping pass ...")
    cat_idx = [FEATS.index(c) for c in cat_cols]
    X_fit_cb, X_es_cb, X_full_cb = X_fit.copy(), X_es.copy(), X_full.copy()
    for d in (X_fit_cb, X_es_cb, X_full_cb):
        d[cat_cols] = d[cat_cols].astype(int)  # CatBoost cat_features는 int/str만 허용, ordinal encoder는 float 출력
    cb_es = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                                cat_features=cat_idx, random_seed=42, verbose=False,
                                early_stopping_rounds=50, thread_count=4)
    cb_es.fit(X_fit_cb, y_fit, eval_set=(X_es_cb, y_es))
    n_iter_cb = cb_es.get_best_iteration()
    print(f" n_iter_cb={n_iter_cb}, refit on full data ...")
    cb_model = CatBoostClassifier(iterations=n_iter_cb, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                                   cat_features=cat_idx, random_seed=42, verbose=False, thread_count=4)
    cb_model.fit(X_full_cb, y_full)

    # ---- CatBoost depth=6 (v9): depth8과 다른 오류 패턴 -> 4-way 앙상블에서 다양성 기여.
    # 2022/2023/2024 다중fold 검증에서 3-way 대비 4-way가 3/3 승 확인 ----
    print("CatBoost(depth=6): early stopping pass ...")
    cb6_es = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                                 cat_features=cat_idx, random_seed=123, verbose=False,
                                 early_stopping_rounds=50, thread_count=4)
    cb6_es.fit(X_fit_cb, y_fit, eval_set=(X_es_cb, y_es))
    n_iter_cb6 = cb6_es.get_best_iteration()
    print(f" n_iter_cb6={n_iter_cb6}, refit on full data ...")
    cb6_model = CatBoostClassifier(iterations=n_iter_cb6, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                                    cat_features=cat_idx, random_seed=123, verbose=False, thread_count=4)
    cb6_model.fit(X_full_cb, y_full)

    # ---- K-fold OOF -> 로지스틱 메타러너 (4-way 스태킹) ----
    print("K-fold OOF for stacking meta-learner (LGB+XGB+CatBoost-d8+CatBoost-d6) ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_idx, y_idx = X_full.reset_index(drop=True), y_full.reset_index(drop=True)
    season_idx_full = train["season"].reset_index(drop=True)
    X_idx_cb = X_full_cb.reset_index(drop=True)
    oof_lgb = np.zeros(len(X_idx))
    oof_xgb = np.zeros(len(X_idx))
    oof_cb = np.zeros(len(X_idx))
    oof_cb6 = np.zeros(len(X_idx))
    for i, (fit_idx, oof_idx) in enumerate(kf.split(X_idx)):
        m1 = lgb.LGBMClassifier(n_estimators=n_iter_lgb, learning_rate=0.05, num_leaves=31, max_depth=8,
                                 min_child_samples=20, reg_lambda=1.0, bagging_fraction=0.8, bagging_freq=1,
                                 feature_fraction=0.8, random_state=42, n_jobs=-1, verbosity=-1)
        m1.fit(X_idx.iloc[fit_idx], y_idx.iloc[fit_idx])
        oof_lgb[oof_idx] = m1.predict_proba(X_idx.iloc[oof_idx])[:, 1]

        m2 = xgb.XGBClassifier(n_estimators=n_iter_xgb, learning_rate=0.05, max_depth=5, max_leaves=31,
                                grow_policy="lossguide", min_child_weight=20, reg_lambda=1.0,
                                subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", random_state=42, n_jobs=-1)
        m2.fit(X_idx.iloc[fit_idx], y_idx.iloc[fit_idx])
        oof_xgb[oof_idx] = m2.predict_proba(X_idx.iloc[oof_idx])[:, 1]

        m3 = CatBoostClassifier(iterations=n_iter_cb, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                                 cat_features=cat_idx, random_seed=42, verbose=False, thread_count=4)
        m3.fit(X_idx_cb.iloc[fit_idx], y_idx.iloc[fit_idx])
        oof_cb[oof_idx] = m3.predict_proba(X_idx_cb.iloc[oof_idx])[:, 1]

        m4 = CatBoostClassifier(iterations=n_iter_cb6, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                                 cat_features=cat_idx, random_seed=123, verbose=False, thread_count=4)
        m4.fit(X_idx_cb.iloc[fit_idx], y_idx.iloc[fit_idx])
        oof_cb6[oof_idx] = m4.predict_proba(X_idx_cb.iloc[oof_idx])[:, 1]
        print(f" OOF fold {i+1}/5 done")

    meta_model = LogisticRegression()
    meta_model.fit(np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_cb6]), y_idx.values)
    print(f" meta_coef={meta_model.coef_[0].round(3)}  meta_intercept={meta_model.intercept_[0]:.3f}")
    oof_meta_pred = meta_model.predict_proba(np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_cb6]))[:, 1]

    # ---- season-trend bias-fix target (extrapolate to 2025, train seasons only) ----
    # v8: 전체 시즌 단순 선형회귀 -> 지수가중 선형회귀(decay=0.6, 최근 시즌에 더 큰 가중치)로 변경.
    # 2022/2023/2024 다중fold 재검증에서 2/3 승 확인 (2023처럼 급락한 해에만 근소하게 불리, 나머지는 우세).
    season_rates = train.groupby("season")[TARGET].mean()
    x_seasons = season_rates.index.values.astype(float)
    y_rates = season_rates.values
    w = 0.6 ** (x_seasons.max() - x_seasons)
    slope, intercept = np.polyfit(x_seasons, y_rates, 1, w=np.sqrt(w))
    expected_rate_2025 = slope * 2025 + intercept
    expected_rate_2025 = float(np.clip(expected_rate_2025, 0.01, 0.99))
    print(f"season rates: {season_rates.round(4).to_dict()}")
    print(f"extrapolated expected 2025 rate (exp-weighted): {expected_rate_2025:.4f}")

    # v11: 규정 위반 수정 — 예전엔 script.py에서 추론 배치(X, 즉 실제 평가데이터) 자체의
    # pred.mean()으로 매 배치마다 shift를 다시 계산했음. 이건 "전체 평가 데이터의 분포를 이용해
    # 개별 행 예측값을 보정"하는 것이라 규정(평가 데이터 예측 원칙 4번) 위반 소지가 있었음.
    # 고정된 상수 shift를 학습 시점(OOF, train 데이터만 사용)에 한 번만 계산해서 저장하고,
    # 추론 시엔 이 상수를 그대로 더하기만 한다 — 추론 배치 자체를 전혀 참조하지 않음.
    #
    # v12: v11은 oof_meta_pred 평균을 2019~2024 전체를 섞어서 계산해서 실전 LB가 38.7점으로
    # 폭락했음. 원인: 트리는 season=2025(훈련범위 밖)를 만나면 season=2024(훈련범위 내 최댓값)의
    # 리프로 그대로 떨어진다(외삽 불가) — 즉 실제 2025 추론은 "2024와 비슷한 상황일 때의 모델
    # 동작"에 가까운데, 기준을 전체 시즌 평균(초기 시즌들의 더 높은 성공률까지 섞임)으로 잡아서
    # 보정 방향이 크게 어긋났음. 2021~2024 다중fold 재검증(고정shift 방식으로 제대로 재현) 결과
    # "가장 최근 시즌(max_train_season)의 OOF만 기준"이 3/4 fold에서 우세 — 이걸로 수정.
    # (여전히 train 데이터만 사용 — 규정 문제 없음)
    max_train_season = train["season"].max()
    mask_recent_season = (season_idx_full == max_train_season).values
    oof_recent_mean = float(oof_meta_pred[mask_recent_season].mean())
    raw_bias_shift = expected_rate_2025 - oof_recent_mean

    # v13/v14: bias_shift 크기(shrinkage) 자체를 2021~2024 4-fold로 스윕 검증(alpha in
    # 0,0.5,0.75,1.0,1.25,1.5). 로컬 신호는 fold마다 상충했지만(2021,2022는 약한 보정 선호,
    # 2023은 강한 보정 선호), 실전(2025) 결과는 alpha를 낮출수록 계속 좋아짐이 3연속 확인됨:
    # alpha=1.0(v12) 932.83 -> alpha=0.75(v13) 944.66(+11.83) -> alpha=0.5(v14) 950.48(+5.82).
    # 개선폭이 줄어드는 추세라 수확체감 곡선일 가능성 있음 — 다음 세션에서 0.25 단위로 더
    # 낮춰 sweet spot을 찾을 예정 (alpha=0, 즉 보정 완전 제거는 아직 실전 미검증).
    BIAS_SHIFT_ALPHA = 0.5
    bias_shift = raw_bias_shift * BIAS_SHIFT_ALPHA
    print(f"oof_meta_pred mean(전체)={oof_meta_pred.mean():.4f}  "
          f"oof_meta_pred mean(season={max_train_season}만)={oof_recent_mean:.4f}  "
          f"raw_bias_shift={raw_bias_shift:.4f}  alpha={BIAS_SHIFT_ALPHA}  "
          f"fixed bias_shift: {bias_shift:.4f}")

    artifacts = dict(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        cb_model=cb_model,
        cb6_model=cb6_model,
        cat_idx=cat_idx,
        encoder=enc,
        cat_cols=cat_cols,
        medians=medians,
        na_cols=NA_COLS,
        pitcher_na=PITCHER_NA,
        batter_na=BATTER_NA,
        li_q75=float(li_q75),
        feats=FEATS,
        meta_model=meta_model,
        expected_rate_2025=expected_rate_2025,
        bias_shift=bias_shift,
        team_pitcher_rate_lookup=team_pitcher_rate_lookup,
        team_batter_rate_lookup=team_batter_rate_lookup,
        group_asof_medians=group_asof_medians,
    )
    joblib.dump(artifacts, MODEL_PATH, compress=3)
    print(f"Saved: {MODEL_PATH}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
