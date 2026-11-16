# الخطوة الرابعة — بناء وتدريب النماذج

## المدخلات
- `../data/model_data.csv` (ناتج الخطوة الثانية)
- `../data/raw/studentRegistration.csv` (ملف OULAD الخام، مو مرفوع على GitHub؛ حطه في `data/raw/` مثل باقي ملفات OULAD. يُستخدم فقط لحذف الطلاب اللي انسحبوا قبل اليوم 28)

## التشغيل
```bash
cd Modeling
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train_models.py
```
خيارات: `--data` مسار ملف آخر، `--registration` مسار ملف التسجيل، `--keep-early-withdrawals` لإبقاء المنسحبين المبكرين.
مدة التشغيل تقريبًا 5 دقائق.

## الهدف (تصنيف ثنائي)
- `at_risk = 1` ← Fail أو Withdrawn
- `at_risk = 0` ← Pass أو Distinction

## الخصائص المسموح بها (معروفة في اليوم 28)
- فئوية: code_module، presentation_start (B/J)، gender، region، highest_education، imd_band، age_band، disability
- رقمية: num_of_prev_attempts، studied_credits، total_clicks_first_28_days، active_days_first_28_days، unique_sites_first_28_days
- ممنوعة: id_student، final_result، date_unregistration، الدرجات وأي شيء بعد اليوم 28، وسنة الفصل.

## منع التسرب
1. حذف 5040 طالب انسحبوا قبل اليوم 28 (معروف مصيرهم مسبقًا).
2. التقسيم 80/20 مقسّم حسب الطالب (StratifiedGroupKFold) فلا يظهر نفس الطالب في التدريب والاختبار.
3. المعالجة (تعويض القيم الناقصة، One-Hot، Scaling) داخل Pipeline وتتعلم من بيانات التدريب فقط.
4. ضبط المعاملات (GridSearchCV) على التدريب فقط، والاختبار يُستخدم مرة واحدة.

## النتائج (بيانات الاختبار، 5510 صف)
| النموذج | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.668 | 0.615 | 0.667 | 0.640 | 0.735 |
| Decision Tree | 0.654 | 0.597 | 0.662 | 0.628 | 0.709 |
| Random Forest | 0.668 | 0.626 | 0.617 | 0.621 | 0.730 |

## المخرجات (للشخص الخامس)
- `models/*.joblib` — Pipeline كامل جاهز للتنبؤ: `joblib.load(...).predict(df)`
- `outputs/test_predictions.csv` — y_true + التنبؤ والاحتمال لكل نموذج
- `outputs/model_comparison.csv` — المقاييس وأفضل المعاملات
- `outputs/feature_importance.csv` — أهمية الخصائص / المعاملات
- `outputs/split.csv` — كل صف تدريب أو اختبار
- `outputs/run_info.json` — الإعدادات والإصدارات
