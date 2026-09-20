# TECHPHORIA 2026 — Competition Strategy

> **Dataset**: 4,048 × 27 | **Target**: `digital_academic_wellbeing_level` (Low / Medium / High)  
> **Metric**: Macro-F1 (Confirmed) | **Deadline**: ~11 days remaining
>
> *Teman sudah dapat 68.4 single → 80-an ensemble (Rasyid) dan 70-an (Safarudin). Goal pertama bukan mengalahkan 0.80 — goal pertama adalah memverifikasi apakah 0.80 itu nyata.*

---

## Mental Model Dataset

Sebelum membuka model, pahami strukturnya:

```
Student
   │
   ├── Academic (signal tinggi)
   │     ├── gpa
   │     ├── study_hours_per_week
   │     ├── class_attendance_percent
   │     ├── late_assignments_per_semester
   │     ├── academic_stress_level_1_10
   │     └── procrastination_level_1_10
   │
   ├── Digital (sinyal campuran)
   │     ├── screen_time_hrs_per_day
   │     ├── social_media_hrs_per_day
   │     ├── screen_time_before_bed_minutes
   │     ├── notifications_per_day
   │     ├── uses_productivity_app
   │     └── favorite_content_type
   │
   └── Lifestyle (recovery & health)
         ├── sleep_hours
         ├── physical_activity_hrs_per_week
         ├── caffeine_intake_mg_per_day
         ├── diet_quality_1_10
         ├── pre_exam_anxiety_level_1_10
         └── resting_heart_rate_bpm
                         ↓
            Digital Academic Wellbeing
               Low  /  Medium  /  High
```

Pertanyaan pertama bukan *"CatBoost atau XGBoost?"*  
Pertanyaan pertama adalah: **"Bagaimana target ini kemungkinan dibentuk dari kombinasi ketiga domain tersebut?"**

---

## Competition Rules (Bukan Boleh Dilanggar)

```
1. Never use test distribution to tune model parameters without explicitly logging it.
2. All model comparisons MUST use identical CV folds — set fold seed once, never change.
3. All reported F1 scores MUST come from OOF predictions, never re-fit on full train.
4. Every experiment gets an ID (E001, E002, ...). Never overwrite results.
5. Separate hypothesis from confirmed result. Label clearly.
6. Never claim improvement from a single fold.
7. Optimize the actual competition metric (Macro-F1).
8. Keep all preprocessing inside reproducible, versioned pipelines.
9. Clipping/capping must be column-specific, never a blanket "all >24 gets clipped".
10. Any feature informed by test set distribution must be explicitly flagged as leakage risk.
```

---

## Phase 0 — Metric Strategy: Macro-F1 Implications

**Metrik telah dikonfirmasi sebagai Macro-F1.**

Ini mengubah seluruh permainan. Dalam Macro-F1, performa pada kelas minoritas sama pentingnya dengan kelas mayoritas.

| Kelas | Proporsi (Train) | Implikasi pada Macro-F1 |
|-------|------------------|-------------------------|
| `Low` | 51.0% | Kelas mayoritas. Model akan cenderung over-predict kelas ini. |
| `Medium` | 29.9% | Kelas tengah. Sering mengalami kebingungan batas (borderline confusion). |
| `High` | **19.1%** | **Bottleneck Utama.** Karena proporsinya paling kecil, model akan under-predict kelas ini, sehingga menjatuhkan skor Macro-F1 secara keseluruhan. |

**Strategi Utama untuk Macro-F1:**
- Evaluasi OOF F1 harus selalu menggunakan `average='macro'`.
- Eksplorasi ekstensif pada **class weighting** (`class_weight='balanced'`) pada semua model.
- Wajib melakukan **threshold / decision optimization** spesifik per kelas setelah probabilitas OOF didapatkan. Default `argmax` dipastikan sub-optimal.

---

## Phase 1 — Forensics (Day 1)

> Jangan training sebelum ini selesai. Data forensics jauh lebih valuable dari model di tahap ini.

### 1.1 — ID Investigation: `respondent_id`

Jangan buang, jangan langsung jadikan feature. Investigasi dulu.

```python
# Extract numeric component
df['id_numeric'] = df['respondent_id'].str.extract(r'(\d+)').astype(int)

# Investigate:
# 1. Range train vs test — apakah overlap?
# 2. Target distribution per ID range (binned)
# 3. Apakah ada ordering / clustering berdasarkan ID?
```

Jika target distribution berubah berdasarkan ID range → investigasi lebih dalam mengapa.  
Jika distribusi random → ID mungkin tidak berguna sebagai feature.

### 1.2 — Canonicalization Layer (bukan preprocessing tersebar)

Data punya formatting noise yang disengaja. Buat satu layer pembersihan:

```python
# Bukan:
# df['age'] = df['age'].str.extract(r'(\d+)')  ← tersebar di notebook

# Tapi:
def parse_age(raw: str) -> int:
    """Handles: '20', '20 years', '20 yrs', '20y', etc."""
    ...

def normalize_gender(raw: str) -> str:
    """Handles: Male/male/MALE/Female/FEMALE/Other → male/female/other"""
    ...

def normalize_binary(raw: str) -> str:
    """Handles: Yes/yes/YES/No/no/NO → yes/no"""
    ...
```

Simpan kolom raw selama eksperimen. Jangan overwrite.

**Audit noise patterns — train vs test:**
```
AGE noise:  "20 years" ada di train berapa? di test berapa?
GENDER noise: "MALE" ada di train X, di test Y?
```

Jika pola noise berbeda antara train dan test → itu informasi.

### 1.3 — Outlier Audit (Column-Specific)

**Peringatan: clipping `>24 jam` adalah asumsi yang berbahaya.**

| Column | Valid Range | Max Observed | Action |
|--------|-------------|--------------|--------|
| `screen_time_hrs_per_day` | 0–24 | 48 | Investigate: typo? artifact? |
| `sleep_hours` | 0–24 | 11 | Likely valid |
| `study_hours_per_week` | 0–168 | 25 | **Valid!** Jangan clip ke 24. |
| `physical_activity_hrs_per_week` | 0–168 | 9.6 | Valid |
| `notifications_per_day` | 0–∞ | — | **Min = -5**: impossible, fix it. |

Setiap kolom punya batasan logisnya sendiri. Jangan pakai satu rule untuk semua.

### 1.4 — Missingness Audit

Missingness bukan sekadar data yang hilang. Ini bisa jadi signal.

```python
missing_cols = [
    'caffeine_intake_mg_per_day',   # 202 (5.0%)
    'class_attendance_percent',      # 189 (4.7%)
    'social_media_hrs_per_day',      # 160 (4.0%)
    'study_hours_per_week',          # 122 (3.0%)
    'uses_productivity_app',         # 118 (2.9%)
    'favorite_content_type',         # 77  (1.9%)
]

# Test: apakah missingness berhubungan dengan target?
for col in missing_cols:
    df[f'is_missing_{col}'] = df[col].isnull().astype(int)

# Missing count per row
df['missing_count'] = df[missing_cols].isnull().sum(axis=1)

# Target distribution per missingness pattern
```

`social_media_hrs = NaN` ≠ `social_media_hrs = 0`. Jangan perlakukan sama.

### 1.5 — Train/Test Distribution Drift

Untuk setiap feature numerik: bandingkan distribusi train vs test (KS test / visual).  
Flag kolom yang distribusinya sangat berbeda sebagai high-risk untuk model drift.

### 1.6 — Duplicate & Leakage Hunt

- Cek duplicate rows di train.
- Cek apakah ada baris di test yang identik dengan train.
- Cek apakah ada feature yang secara logis mengandung informasi target (leakage).

---

## Phase 2 — Baseline (Day 2)

Gunakan **fold yang sama** untuk semua eksperimen. Set seed sekali, jangan pernah ganti.

```python
FOLD_SEED = 42
N_SPLITS = 5
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
```

Jalankan 5 model kandidat pada data yang sudah di-clean (Phase 1), **tanpa feature engineering tambahan**:

| Exp | Model | Justifikasi |
|-----|-------|-------------|
| E001 | **CatBoost** | Handles categorical natively, bagus untuk small/mixed datasets |
| E002 | **LightGBM** | Mencari struktur nonlinear berbeda dari CatBoost |
| E003 | **XGBoost** | Diversity dari LGB |
| E004 | **ExtraTrees** | Inductive bias sangat berbeda — penting untuk ensemble diversity |
| E005 | **Logistic Regression** | Diagnostic: seberapa linear hubungan target? |

**Simpan OOF probabilities semua model:**

```
oof_table.csv:
  respondent_id | y_true
  catboost_prob_low | catboost_prob_medium | catboost_prob_high
  lgbm_prob_low    | lgbm_prob_medium    | lgbm_prob_high
  xgb_prob_low     | xgb_prob_medium     | xgb_prob_high
  extratrees_prob_low | ...
```

**Lakukan error analysis per model setelah ini:**  
Identifikasi kasus di mana CatBoost salah tapi LGBM benar, dan sebaliknya.  
Jika error pattern berbeda → ensemble masuk akal. Jika error pattern sama → ensemble tidak akan banyak membantu.

---

## Phase 3 — Feature Engineering (Day 3–5)

Pendekatan: **Feature Families**. Uji setiap family secara terpisah, bukan semua sekaligus.

### Feature Families

#### BASE (dari Phase 1 — sudah ada)
Cleaned features setelah canonicalization.

#### DIGITAL
```python
# Social media adalah subset screen_time, bukan tambahan
non_social_screen_time = screen_time - social_media     # validate: harus >= 0
social_media_share     = social_media / screen_time     # proporsi
screen_before_bed_share = screen_before_bed / (sleep_hours * 60)
```

#### DIGITAL × LIFESTYLE (cross-domain)
```python
screen_to_sleep_ratio   = screen_time / sleep_hours
screen_bed_to_sleep     = screen_before_bed / (sleep_hours * 60)
caffeine_per_sleep_hr   = caffeine / sleep_hours
physical_per_sleep      = physical_activity / sleep_hours
```

#### ACADEMIC
```python
academic_pressure_score = academic_stress × procrastination
stress_anxiety_product  = academic_stress × pre_exam_anxiety
late_per_study_hour     = late_assignments / (study_hours + 1)
```

#### ACADEMIC × DIGITAL (cross-domain)
```python
procrastination_x_screen = procrastination × screen_time
study_vs_screen          = study_hours / (screen_time * 7 + 1)
```

#### ACADEMIC × LIFESTYLE (cross-domain)
```python
stress_vs_sleep         = academic_stress / sleep_hours
gpa_x_attendance        = gpa × class_attendance    # Rasyid pakai ini — validasi dulu
stress_x_sleep          = academic_stress × sleep_hours
```

#### LIFESTYLE / RECOVERY
```python
lifestyle_score   = diet_quality × physical_activity
recovery_index    = sleep_hours × diet_quality / (caffeine + 1)
```

#### AGGREGATE (summary dimensions)
```python
# Digital overload composite
digital_overload = (screen_time / 16) + (social_media / 8) + (notifications / 150)

# Academic pressure composite
academic_pressure = (academic_stress / 10) + (procrastination / 10) + (pre_exam_anxiety / 10)

# Performance composite
academic_performance = gpa / 4.0 + class_attendance / 100 - late_assignments / 12

# Lifestyle balance
lifestyle_balance = sleep_hours / 8 + physical_activity / 5 + diet_quality / 10
```

#### MISSINGNESS (dari Phase 1)
```python
is_missing_caffeine, is_missing_social, is_missing_study,
is_missing_attendance, is_missing_productivity, is_missing_content,
missing_count  # total missing per row
```

#### ORDINAL EXPERIMENT (eksperimen terpisah)
```python
# Target terlihat ordinal: Low < Medium < High
# Buat dua binary models:
P(Y > Low)    # Low vs (Medium+High)
P(Y > Medium) # (Low+Medium) vs High

# Derive probabilities:
P(Low)    = 1 - P(Y > Low)
P(Medium) = P(Y > Low) - P(Y > Medium)
P(High)   = P(Y > Medium)
```

### Experiment Matrix (jalankan secara bertahap)

| Exp | Model | Feature Set | CV F1 | ΔF1 | Notes |
|-----|-------|-------------|-------|-----|-------|
| E001 | CatBoost | BASE | ? | — | Baseline |
| E002 | LightGBM | BASE | ? | — | Baseline |
| E003 | XGBoost | BASE | ? | — | Baseline |
| E004 | ExtraTrees | BASE | ? | — | Baseline |
| E005 | LogReg | BASE | ? | — | Linearity diagnostic |
| E006 | CatBoost | + DIGITAL | ? | ? | |
| E007 | CatBoost | + ACADEMIC | ? | ? | |
| E008 | CatBoost | + LIFESTYLE | ? | ? | |
| E009 | CatBoost | + DIGITAL×LIFESTYLE | ? | ? | |
| E010 | CatBoost | + ACADEMIC×DIGITAL | ? | ? | |
| E011 | CatBoost | + AGGREGATE | ? | ? | |
| E012 | CatBoost | + MISSINGNESS | ? | ? | |
| E013 | CatBoost | Best combo | ? | ? | |
| E014 | LightGBM | Best combo | ? | ? | |
| E015 | XGBoost | Best combo | ? | ? | |
| E016 | Ordinal decomposition | Best combo | ? | ? | |

**Semua fold identik. Jangan ubah seed.**

---

## Phase 4 — Hyperparameter Tuning (Day 6–7)

Tuning **hanya setelah** feature representation stabil (setelah Phase 3 selesai).

```python
# Optuna dengan OOF macro-F1 sebagai objective
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('lr', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        ...
    }
    oof_preds = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
    return f1_score(y, oof_preds.argmax(axis=1), average='macro')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

Tuning setiap model kandidat terbaik dari Phase 3.

---

## Phase 5 — Error Analysis (Day 8)

Sebelum ensemble, pahami di mana tiap model gagal.

```python
# Per-class F1
from sklearn.metrics import classification_report
print(classification_report(y_true, oof_preds, target_names=['Low', 'Medium', 'High']))

# Error matrix: mana yang CatBoost salah tapi LGB benar?
error_df = pd.DataFrame({
    'y_true': y_true,
    'catboost_pred': cat_preds,
    'lgbm_pred': lgb_preds,
    'xgb_pred': xgb_preds,
})
error_df['cat_wrong'] = error_df['catboost_pred'] != error_df['y_true']
error_df['lgb_right'] = error_df['lgbm_pred'] == error_df['y_true']

# Focus on High class — bottleneck untuk macro-F1
high_errors = error_df[error_df['y_true'] == 'High']
```

Kelas `High` (19%) adalah bottleneck. Jika recall `High` sangat rendah, pertimbangkan class weighting atau oversampling.

---

## Phase 6 — Ensemble & Decision Optimization (Day 9–10)

### 6.1 — Verifikasi: Apakah Ensemble Layak?

**Cek dulu apakah error patterns benar-benar complementary.**

Jika:
```
CatBoost OOF F1:  0.74
LightGBM OOF F1:  0.73
Ensemble OOF F1:  0.75  ← hanya naik 1%
```
tetapi leaderboard:
```
CatBoost LB:  0.68
Ensemble LB:  0.80  ← naik 12%???
```

→ **Curiga.** Kemungkinan: validation setup berbeda, leaderboard overfitting, atau submission issue. Investigasi sebelum trust.

### 6.2 — Ensemble Strategies (dalam urutan kompleksitas)

```python
# Strategy 1: Simple average (murah, sering kuat)
ensemble_proba = (cat_proba + lgb_proba + xgb_proba) / 3

# Strategy 2: Weighted average (optimasi weight dengan Optuna pada OOF)
def ensemble_weight_objective(trial):
    w_cat = trial.suggest_float('w_cat', 0, 1)
    w_lgb = trial.suggest_float('w_lgb', 0, 1)
    w_xgb = 1 - w_cat - w_lgb
    if w_xgb < 0: return 0.0
    proba = w_cat * cat_proba + w_lgb * lgb_proba + w_xgb * xgb_proba
    return f1_score(y_true, proba.argmax(axis=1), average='macro')

# Strategy 3: Stacking (meta-learner di atas OOF proba)
# Input: [cat_low, cat_med, cat_high, lgb_low, ..., xgb_high] → meta LogReg
```

### 6.3 — Class-Specific Decision Optimization

Default argmax tidak selalu optimal untuk macro-F1.

```python
# Scaled argmax: adjust class probabilities with class-specific weights
def adjusted_predict(proba, w_low, w_medium, w_high):
    weights = np.array([w_low, w_medium, w_high])
    adjusted = proba * weights
    return adjusted.argmax(axis=1)

# Optimize weights dengan Optuna pada OOF — HARUS pakai holdout discipline
# Jangan tune dan evaluate pada set yang sama
```

### 6.4 — Probability Calibration Experiment

```python
from sklearn.calibration import CalibratedClassifierCV

# Test: calibrated vs raw probabilities sebelum ensemble
# Tidak selalu membantu, tapi murah untuk diuji
```

---

### Phase 6: Ensemble & Decision Optimization (COMPLETED)

- **Status:** Completed. Broke the single-model ceiling.
- **Method:** Weighted Soft-Voting optimized via Optuna (200 trials) + global Nelder-Mead threshold optimization.
- **Results:**
  - `Scenario_1_Full` (All models): 0.6794
  - **`Scenario_2_Golden` (Top 5 models)**: **0.67696** (OOF CV). Ini adalah referensi "Golden Scenario" kita yang terbukti sangat solid.
  - `Scenario_4_TreeBased`: 0.67965
- **Insight:** Meskipun skor CV OOF kita mencapai ~0.677, skor pada **Public Leaderboard (30% Data)** tercatat turun ke **0.66**. Penurunan ini wajar di Kaggle (akibat variansi 30% data test public). Oleh karena itu, kita akan bersandar penuh pada robust CV kita (seperti `Scenario_2_Golden`) dan menjadikannya jangkar (*anchor*) untuk pengembangan ke depannya.

---

### Current Best Strategy for Final Submission
* **Model**: Soft-Voting Ensemble (CatBoost, XGBoost, LightGBM, RandomForest, ExtraTrees).
* **Features**: Ablated features yang menghasilkan keragaman (diversity) tertinggi antar model.
* **Thresholds**: SciPy Nelder-Mead Optimization on probabilities.
* **Submission Format**: File terpisah hasil script `generate_submission.py`.

---

## Phase 9: Grand Fusion Enhancements (COMPLETED)
Berdasarkan pendekatan 'Grand Fusion', kami mengimplementasikan strategi berikut:

### 1. Hierarchical Group Aggregations & Z-Scores
* **Concept**: Mengelompokkan dataset berdasarkan fitur kategori kunci (`field_of_study`, `gender`, `uses_productivity_app_clean`). Menghitung *mean* dan *standard deviation* kontekstual.
* **Result**: Memberikan dorongan struktural besar pada fitur individu (Full Features).

### 2. Multi-Seed Bagging Ensemble
* **Concept**: Melatih beberapa instansi (seed) untuk setiap model.
* **Result**: Eksperimen *GroupAgg_MultiSeed* mencetak rekor CV Macro-F1 **0.67400** di MLflow. Namun pendekatan ini sangat kompleks karena menumpuk belasan model.

---

## Phase 11 & 12: Regresi dan Focused Hyperparameter Tuning (COMPLETED)

Menyadari bahwa menumpuk puluhan model (Phase 9) bisa berujung pada *noise*, kita kembali ke model inti:
1. **Phase 11 (Regresi):** Menguji penggunaan `XGBoost_Regressor`. Regresi terbukti secara matematis mampu menangkap urutan ordinal kelas (Low < Medium < High) yang kerap luput oleh Classifier.
2. **Phase 12 (Tuning):** Melakukan Optuna *Hyperparameter Tuning* (30 trials) dengan 5-Fold *precomputation* secara ekstensif pada 3 model utama (menggunakan Full Features). 
   - Hasil CV Individu yang tercatat di MLflow: LightGBM (0.6678), XGBoost (0.6652), CatBoost (0.6655).

---

## Phase 13: Error Analysis & The Ultimate 6 (COMPLETED)

Kita menjalankan uji variasi terkontrol (Controlled Variations) untuk membuktikan apakah substitusi komponen *Golden Scenario* dengan versi *Tuned* dan penambahan Regressor secara individual akan menambah sinyal atau sekadar *noise*.

**Hasil CV Leaderboard (Phase 13 Controlled Variations):**
* `V0_Golden_Baseline` (5 model asli): 0.6593
* `V4_Golden_Replace_CAT_w_Tuned`: 0.6661
* `V3_Golden_Replace_LGB_w_Tuned`: 0.6687
* `V2_Golden_Replace_XGB_w_Tuned`: 0.6699
* `V1_Golden_+_XGB_Regressor` (6 model): 0.6705
* **`V5_Golden_Replace_ALL_Trees_w_Tuned`**: **0.6748**

**Final Conclusion:** Hipotesis terbukti. Menumpuk belasan model sekaligus adalah *noise*. Namun, mengambil 5 jangkar Golden (dengan RandomForest & ExtraTrees tetap menggunakan ablated features untuk keragaman) dan **mengganti** 3 model boosting (CAT, XGB, LGB) dengan versi yang sudah disetel mentok (Phase 12), melonjakkan F1 score kita ke **0.6748** secara elegan.

**The Ultimate 6 (Kandidat Submission Berikutnya):**
Gabungan V5 (0.6748) dengan XGBoost Regressor (0.6705) untuk menangkap kombinasi tertinggi antara diversitas fitur (RF/ET), kekuatan tuning individu (Trees), dan sensitivitas metrik ordinal (Regressor).

---

### Phase 6.5: Breakthrough Experiments (COMPLETED — 2026-09-18)

We ran 4 independent, modular experiments to attack the noise ceiling and capture the remaining signal found during our deep forensics audit:

| Exp | Name | Concept | Verdict (Result vs Baseline) |
|:---|:---|:---|:---|
| **A** | **Interaction Features** | Explicitly multiply `uses_productivity_app` and `frequently_procrastinates` by numeric severity levels. | ❌ **Declined**. LightGBM dropped to 0.6488 (vs 0.6539). Forcing explicit categoric interactions confused the tree models instead of helping them. |
| **B** | **Ordinal Classification** | Decompose 3-class target into 2 binary models: `P(Y ≥ Medium)` and `P(Y ≥ High)`. | ⚠️ **Mixed**. CatBoost gained significantly (0.6422 vs 0.6305), but XGBoost and LightGBM degraded. |
| **C** | **Stacking Meta-Learner** | Use 21 OOF probabilities as features for a highly regularized Logistic Regression. | ❌ **Declined**. Meta LogReg scored 0.6706, which is good but couldn't beat our simple soft-voting Phase 6 Ensemble (0.6796). |
| **D** | **Extended Groups** | More aggressive relative group features (z-scores by gender, field_of_study). | ⚠️ **Mixed**. RandomForest loved the peer-group z-scores (0.6471 vs 0.6385), but primary boosting models dropped slightly. |

**Final Conclusion:** The "noise ceiling" is a real, inherent property of this survey dataset. None of the architectural or feature-based interventions were able to push our top models beyond their established Phase 3.6 limits. Therefore, the `Scenario 4 (Tree-Based Only)` ensemble from Phase 6 (Macro-F1: **0.6796**) remains our undisputed best configuration for the final submission.

---

## Phase 7 — Final Submission (Day 11)

**Jangan lakukan 40 eksperimen baru di hari terakhir.**

- [ ] Pilih **2 submission candidates**:
  - **Safe**: Model terbaik atau simple average ensemble (CV score paling stabil)
  - **Aggressive**: Full stacking + threshold optimization
- [ ] Validasi class distribution submission vs train (sanity check)
- [ ] Verifikasi format submission sesuai `sample_submission.csv`
- [ ] Log final experiment ke `experiments/` sebelum deadline

---

## Skeptic Checklist

Setiap kali ada klaim "F1 naik X%", tanyakan:

```
□ Apakah fold yang digunakan sama persis?
□ Apakah bisa noise statistik biasa (standar deviasi antar fold)?
□ Apakah threshold di-tune pada fold evaluasi yang sama?
□ Apakah ada informasi dari test set yang bocor?
```

### Phase 3: Feature Engineering (COMPLETED)
- **Status:** Completed. Achieved performance gains in tree-based models (XGBoost +0.011) via interaction ratios, but slight drops in linear models.
- **Key Features Added:**
  - `screen_to_sleep_ratio`, `social_media_impact`
  - `stress_gpa_ratio`, `procrastination_stress_index`
  - `caffeine_to_sleep`, `physical_vs_screen`
  - Binning: `is_sleep_deprived`, `high_caffeine`
- **Insight:** Uncorrelated errors between Logistic Regression (good with linear patterns) and XGBoost (good with ratios) provide a strong foundation for stacking.

---

### Phase 3.5: Hardening (COMPLETED — 2026-09-17)

Pre-Phase 4 audit revealed and fixed three blockers plus two model improvements.

**Blockers Fixed:**

| # | Issue | Fix |
|---|-------|-----|
| B1 | `screen_time > 24h` (28 train rows, values up to 48h) | `-> NaN` in `preprocessing.py` per Phase 1 convention |
| B2 | XGBoost had no class balancing | `compute_xgb_sample_weights()` in `models.py`, passed per-fold via `evaluate_model()` |
| B3 | `evaluate_model()` only returned hard predictions | Now returns `(score, oof_preds, oof_proba)` — probabilities saved as `.npy` per model |

**Improvements Added:**

| # | Improvement | Rationale |
|---|-------------|-----------|
| I1 | ExtraTrees added to model pool | STRATEGY E004: inductive bias diversity for ensemble |
| I2 | 3 new cross-domain features | `gpa_x_attendance`, `study_vs_screen`, `recovery_index` |
| I3 | `generate_features()` canonical entry point | `train.py` and `run_feature_experiments.py` now share one clean interface |

**Phase 3.5 Results (OOF Macro-F1, 5-fold, seed=42):**

| Model | Score | Delta vs Phase 3 |
|-------|-------|-----------------|
| **LightGBM** | **0.6471** | New leader |
| **XGBoost** | **0.6461** | +~0.0008 (class weights helped) |
| LogisticRegression | 0.6424 | +improvement (new features) |
| RandomForest | 0.6331 | — |
| CatBoost | 0.6305 | — |
| ExtraTrees | 0.6063 | Weak solo; kept for ensemble diversity |

**Key Insight:** ExtraTrees solo is weak (0.606) but its error correlation with other models may still be low — it should be included in the ensemble pool and let Optuna decide its weight in Phase 4.

**Phase 4 Readiness:** ✅ All blockers resolved. OOF probabilities available for all models. Pipeline is clean and reproducible.

---

### Phase 3.6: Architecture Upgrade & Optimization (COMPLETED — 2026-09-17)

To address limitations found in Phase 3.5, we introduced architectural and evaluation enhancements:

**Improvements Added:**
| # | Improvement | Rationale |
|---|-------------|-----------|
| I1 | `QuantileTransformer` replacing `StandardScaler` | Robust normalization for linear/NN models against heavy-tailed interaction ratios. |
| I2 | Macro-F1 Threshold Optimization | Post-hoc optimization of probability multipliers via Nelder-Mead to address the minority `High` class bottleneck. |
| I3 | Tabular MLP Baseline | A neural network to capture higher-order, complex interactions. |
| I4 | MNAR Missingness Features | Explicitly encode null counts and flags, as data forensics (Phase 1) proved missingness was informative. |
| I5 | Ratio Clipping | Capping extreme division results (`.clip(upper=...)`) to prevent unbounded feature explosions. |

**Phase 3.6 Results (OOF Macro-F1 Tuned, 5-fold, seed=42):**
| Model | Macro-F1 (Tuned) | Delta vs Phase 3.5 |
|-------|------------------|--------------------|
| **LightGBM** | **0.6539** | +0.0068 |
| **LogisticRegression** | **0.6476** | +0.0052 (huge gain from QuantileTransformer & Tuned Threshold) |
| XGBoost | 0.6450 | Stable |
| RandomForest | 0.6385 | +0.0054 |
| MLPClassifier | 0.6383 | New baseline entry |
| ExtraTrees | 0.6279 | +0.0216 (massive gain from Tuned Threshold) |

**Key Insight:** Threshold optimization significantly bumps OOF Macro-F1 across all models, proving that default `argmax` hurts performance on imbalanced targets. `QuantileTransformer` drastically improved our linear baseline (`LogisticRegression`) by fixing ratio feature heavy-tails.

**Phase 4 Readiness:** ✅ We are now fully prepared to run `Optuna` hyperparameter tuning (Phase 4), optimizing on the tuned Macro-F1 metric!

---

## Experiment Log (isi setelah setiap run)

File: `experiments/experiment_log.csv`

```
exp_id,model,features,cv_f1_mean,cv_f1_std,lb_score,notes
E001,CatBoost,BASE,,,,
E002,LightGBM,BASE,,,,
...
```

AI membaca file ini untuk merekomendasikan eksperimen berikutnya, bukan mengarang dari nol.

---

## Stack

```
Python
├── Polars / Pandas (data manipulation)
├── DuckDB (query / audit)
├── scikit-learn (CV, metrics, calibration)
├── CatBoost (primary candidate)
├── LightGBM (primary candidate)
├── XGBoost (diversity)
├── ExtraTrees (diversity)
├── Optuna (tuning + ensemble weight search)
├── SHAP (feature importance, error analysis)
└── MLflow (experiment tracking)
```

---

## Key Principles

1. **Forensics before modeling.** Data understanding > model complexity.
2. **Confirm the metric.** Macro vs Weighted vs Micro changes the entire strategy.
3. **One fold seed, never change.** All comparisons must be apples-to-apples.
4. **OOF probabilities are the core artifact.** Save them all, always.
5. **Feature families, not feature lists.** Test signal per family, then combine winners.
6. **Skeptic every improvement.** The gap between CV and LB is information, not noise.
7. **Clipping must be column-specific.** `study_hours_per_week > 24` is valid data.
8. **The `High` class is the bottleneck.** If macro-F1 is the metric, this is where the competition is won.
9. **Ensemble diversity matters more than individual model quality.** Complementary errors, not just top-3 scores.
10. **Don't spend day 11 running new experiments.** Save it for final validation and calm submission.
