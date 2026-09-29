# Phase 17 Ensemble Error Analysis

## Confusion Matrix Summary
| True   |   High |   Low |   Medium |
|:-------|-------:|------:|---------:|
| High   |    473 |    20 |      280 |
| Low    |     22 |  1608 |      435 |
| Medium |    175 |   262 |      773 |


## Most Common Misclassifications
- **True:Low -> Pred:Medium**: 435 cases (10.7% of total data)
- **True:High -> Pred:Medium**: 280 cases (6.9% of total data)
- **True:Medium -> Pred:Low**: 262 cases (6.5% of total data)
- **True:Medium -> Pred:High**: 175 cases (4.3% of total data)
- **True:Low -> Pred:High**: 22 cases (0.5% of total data)


## Feature Analysis: True:Medium -> Pred:Low vs True:Medium -> Pred:Medium
Features that are statistically different when the model mistakes a Medium for a Low:
| Feature                        |   Correct Mean |   Error (Pred Low) Mean |     p-value |
|:-------------------------------|---------------:|------------------------:|------------:|
| academic_stress_level_1_10     |        5.32859 |                 6.39695 | 2.00819e-07 |
| pre_exam_anxiety_level_1_10    |        3.41656 |                 4.21374 | 4.75813e-07 |
| gpa                            |        2.43591 |                 2.33885 | 6.83776e-07 |
| procrastination_level_1_10     |        5.3674  |                 6.17939 | 1.9683e-06  |
| sleep_hours                    |        6.91759 |                 6.6458  | 0.00108888  |
| screen_time_before_bed_minutes |       33.5627  |                38.3931  | 0.00153848  |
| screen_time_hrs_per_day        |        8.48525 |                 9.32977 | 0.0146343   |
| notifications_per_day          |       74.273   |                83.4237  | 0.0147915   |
| late_assignments_per_semester  |        3.65589 |                 3.99618 | 0.0307476   |


## Feature Analysis: True:Medium -> Pred:High vs True:Medium -> Pred:Medium
Features that are statistically different when the model mistakes a Medium for a High:
| Feature                        |   Correct Mean |   Error (Pred High) Mean |     p-value |
|:-------------------------------|---------------:|-------------------------:|------------:|
| social_media_hrs_per_day       |        3.99251 |                  2.83353 | 6.93668e-09 |
| procrastination_level_1_10     |        5.3674  |                  4.38857 | 3.22923e-07 |
| screen_time_before_bed_minutes |       33.5627  |                 24.6     | 1.3466e-06  |
| gpa                            |        2.43591 |                  2.54371 | 2.9322e-05  |
| sleep_hours                    |        6.91759 |                  7.16686 | 0.00253012  |
| academic_stress_level_1_10     |        5.32859 |                  4.67429 | 0.00653084  |
| resting_heart_rate_bpm         |       91.4541  |                 87.5657  | 0.00761267  |