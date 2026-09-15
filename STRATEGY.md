# Experiment Strategy: Digital Academic Wellbeing

> **Competition Goal**: Maximize **F1-score** on a 3-class classification task  
> **Target**: `digital_academic_wellbeing_level` → `Low` (51%) | `Medium` (30%) | `High` (19%)  
> **Train Size**: 4,048 rows × 25 features (+ id + target)  
> **Test Size**: ~972 rows

---

## Data Profile Summary

### Class Distribution (Imbalanced)
| Class  | Count | Percentage |
|--------|-------|------------|
| Low    | 2,065 | 51.0%      |
| Medium | 1,210 | 29.9%      |
| High   |   773 | 19.1%      |

### Missing Values
| Column                     | Missing | % Missing |
|----------------------------|---------|-----------|
| caffeine_intake_mg_per_day | 202     | 5.0%      |
| class_attendance_percent   | 189     | 4.7%      |
| social_media_hrs_per_day   | 160     | 4.0%      |
| study_hours_per_week       | 122     | 3.0%      |
| uses_productivity_app      | 118     | 2.9%      |
| favorite_content_type      | 77      | 1.9%      |

### Top Correlated Features (with target)
| Feature                        | Correlation |
|--------------------------------|-------------|
| procrastination_level_1_10     | -0.351      |
| sleep_hours                    | +0.317      |
| pre_exam_anxiety_level_1_10    | -0.302      |
| gpa                            | +0.265      |
| academic_stress_level_1_10     | -0.246      |
| social_media_hrs_per_day       | -0.234      |
| screen_time_before_bed_minutes | -0.213      |
| class_attendance_percent       | +0.187      |

### Data Quality Issues Detected
1. **`gender`**: Inconsistent casing → `Male/MALE/male`, `Female/FEMALE/female`, `Other` (7 unique values, should be 3).
2. **`age`**: Stored as `object/string` despite being numeric (72 unique values, range likely 18–60+). Needs type casting.
3. **`notifications_per_day`**: Min value is **-5**, which is physically impossible. Likely a data entry error / outlier.

---

## Experiment Roadmap

### Phase 0 — Baseline & Infrastructure
> Objective: Get a working end-to-end pipeline with a naive model to establish a baseline score.

- [ ] **0.1** Clean `gender` column (normalize casing to 3 categories: `Male`, `Female`, `Other`).
- [ ] **0.2** Convert `age` from string to integer.
- [ ] **0.3** Handle missing values with simple median/mode imputation.
- [ ] **0.4** Label-encode or one-hot-encode all categorical features.
- [ ] **0.5** Train a default `LGBMClassifier` with 5-Fold **Stratified CV**.
- [ ] **0.6** Log baseline CV F1-score (macro) to MLflow/W&B.
- [ ] **0.7** Generate first Kaggle submission.

**Expected Outcome**: A baseline F1-score to benchmark all future improvements against.

---

### Phase 1 — Data Cleaning & Robust Preprocessing
> Objective: Thoroughly clean the data and handle edge cases.

- [ ] **1.1** Fix `notifications_per_day` outlier (`-5` → clip to 0 or treat as missing).
- [ ] **1.2** Cap `screen_time_hrs_per_day` at a reasonable maximum (current max is 48, which is impossible).
- [ ] **1.3** Investigate `age` distribution — values go up to 59 which seems high for students. Consider binning or capping.
- [ ] **1.4** Add a binary `is_missing_*` indicator column for each column that has missing values (missingness itself can be informative).
- [ ] **1.5** Experiment with multiple imputation strategies:
  - Median/mode (baseline).
  - KNN Imputer.
  - Iterative Imputer (MICE).
- [ ] **1.6** Re-run CV, compare F1 against Phase 0 baseline.

---

### Phase 2 — Feature Engineering
> Objective: Create new features that capture meaningful interactions and domain knowledge.

#### 2.1 — Ratio & Interaction Features
- [ ] `social_media_ratio` = `social_media_hrs_per_day / screen_time_hrs_per_day`
- [ ] `study_to_screen_ratio` = `study_hours_per_week / (screen_time_hrs_per_day * 7)`
- [ ] `stress_x_anxiety` = `academic_stress_level_1_10 * pre_exam_anxiety_level_1_10`
- [ ] `sleep_x_stress` = `sleep_hours * academic_stress_level_1_10`
- [ ] `caffeine_per_sleep_hour` = `caffeine_intake_mg_per_day / sleep_hours`
- [ ] `procrastination_x_late` = `procrastination_level_1_10 * late_assignments_per_semester`
- [ ] `attendance_x_gpa` = `class_attendance_percent * gpa`
- [ ] `digital_overload_index` = `screen_time_hrs_per_day + social_media_hrs_per_day + notifications_per_day / 50` (normalized composite)

#### 2.2 — Binning & Encoding
- [ ] Bin `age` into groups: `teen (≤19)`, `young_adult (20–25)`, `adult (26–35)`, `senior (36+)`.
- [ ] Bin `sleep_hours` into clinical categories: `<5 (severely_deprived)`, `5–6 (deprived)`, `6–8 (normal)`, `>8 (excess)`.
- [ ] Bin `caffeine_intake_mg_per_day` into `none`, `low`, `moderate`, `high`.
- [ ] Target-encode high-cardinality categoricals (`field_of_study`) with proper CV-based encoding to avoid leakage.

#### 2.3 — Aggregate / Statistical Features
- [ ] Mean/Std of all `*_level_1_10` columns → `mean_self_reported_level`, `std_self_reported_level`.
- [ ] Count of "positive" binary flags (`uses_productivity_app=Yes`, `takes_extra_tutoring=Yes`, `family_academic_achievement_history=Yes`) → `positive_factor_count`.
- [ ] Count of "negative" indicators (`frequently_procrastinates=Yes`, `major_life_event=Yes`) → `negative_factor_count`.

#### 2.4 — Feature Selection
- [ ] Use SHAP values from the best single model to rank feature importance.
- [ ] Try Recursive Feature Elimination (RFE) or Boruta to prune noise features.
- [ ] Compare CV scores with full features vs. selected features.

---

### Phase 3 — Model Training & Comparison
> Objective: Train multiple model families and find the best single model.

- [ ] **3.1** Train and tune models via Optuna (50–200 trials each):
  - `LGBMClassifier`
  - `XGBClassifier`
  - `CatBoostClassifier`
  - `sklearn.ensemble.HistGradientBoostingClassifier`
  - `sklearn.ensemble.RandomForestClassifier`
  - `sklearn.ensemble.ExtraTreesClassifier`
- [ ] **3.2** For each model, log:
  - CV F1-macro (mean ± std across folds)
  - Per-class F1 scores (to identify which class is hardest)
  - SHAP summary plot
- [ ] **3.3** Evaluate if class weighting (`class_weight='balanced'` or `scale_pos_weight`) improves minority class (`High`) F1.
- [ ] **3.4** Try oversampling (SMOTE/ADASYN) or undersampling techniques if class weights alone aren't enough.

---

### Phase 4 — Advanced Tuning
> Objective: Squeeze out remaining performance from top models.

- [ ] **4.1** Bayesian hyperparameter search with Optuna using the macro-F1 as the objective.
- [ ] **4.2** Tune `threshold` / calibration: Instead of `argmax`, optimize decision boundaries for each class using the OOF predictions.
- [ ] **4.3** Try different CV strategies:
  - `StratifiedKFold` (baseline, k=5 and k=10).
  - `RepeatedStratifiedKFold` (3×5) for more stable estimates.
- [ ] **4.4** Run post-hoc calibration (Platt Scaling / Isotonic Regression) on predicted probabilities.

---

### Phase 5 — Ensembling
> Objective: Combine diverse models for a more robust final prediction.

- [ ] **5.1** Collect OOF (out-of-fold) predictions from top 3–5 models.
- [ ] **5.2** **Soft Voting**: Weighted average of predicted probabilities → `argmax`.
- [ ] **5.3** **Stacking**: Train a meta-learner (Logistic Regression or small LGBM) on the OOF probability matrix.
- [ ] **5.4** **Blending**: Holdout-based blending as an alternative to stacking.
- [ ] **5.5** Optimize ensemble weights using Optuna (maximize macro-F1 on OOF).
- [ ] **5.6** Verify ensemble CV F1 > best single model CV F1.

---

### Phase 6 — Final Submission & Leaderboard Strategy
> Objective: Maximize public + private leaderboard score.

- [ ] **6.1** Check for train-test distribution shift by comparing feature distributions.
- [ ] **6.2** Generate final predictions using the full ensemble pipeline.
- [ ] **6.3** Post-process predictions: Sanity check class distribution of submission vs. train set.
- [ ] **6.4** If allowed 2 submissions:
  - **Sub 1 (Safe)**: Best single model or simple ensemble (high CV correlation).
  - **Sub 2 (Aggressive)**: Full stacking ensemble with threshold optimization.
- [ ] **6.5** Document every experiment and its result in `experiments/` for reproducibility.

---

## Key Principles

1. **Trust your CV, not the public LB**. A well-designed Stratified CV with enough folds will correlate better with the private leaderboard than overfitting to the public LB.
2. **Feature engineering >> Hyperparameter tuning**. Spend 60% of effort on features, 25% on ensembling, 15% on tuning.
3. **Diversity in ensembles matters**. Combine models that make *different* errors, not just the top-3 best models.
4. **Log everything**. Every experiment should be tracked in MLflow/W&B with config, CV score, and artifacts.
5. **Handle class imbalance explicitly**. The `High` class (19%) will be the bottleneck for macro-F1. Focus special attention on improving its recall.
