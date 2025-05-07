### ✅ **Day 5 – Feature Selection & Tuning mô hình ML**

**🎯 Mục tiêu:** Biết cách chọn lựa các đặc trưng (features) quan trọng, tối ưu hóa mô hình với Grid Search và Cross-validation để cải thiện kết quả.

---

#### 1. Feature Importance & Feature Selection

##### 📌 Với mô hình cây (Tree-based models):

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier()
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X_train.columns

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feat_df)
```

**Giải thích:**

*   `RandomForestClassifier`: Mô hình Random Forest.
*   `feature_importances_`: Thuộc tính trả về độ quan trọng của các đặc trưng.
*   `feat_df`: DataFrame chứa tên đặc trưng và độ quan trọng của chúng, được sắp xếp theo thứ tự giảm dần của độ quan trọng.

##### 📌 Lọc theo thống kê (SelectKBest):

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X_train, y_train)

selected_features = X_train.columns[selector.get_support()]
print("Top features:", list(selected_features))
```

**Giải thích:**

*   `SelectKBest`: Lớp để chọn các đặc trưng tốt nhất dựa trên một hàm đánh giá.
*   `f_classif`: Hàm đánh giá sử dụng phân tích phương sai (ANOVA) để tính điểm cho các đặc trưng.
*   `k`: Số lượng đặc trưng tốt nhất cần chọn.
*   `get_support()`: Phương thức trả về một mảng boolean cho biết các đặc trưng nào đã được chọn.

**📚 Tham khảo:**

*   [Scikit-learn documentation - RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
*   [Scikit-learn documentation - SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)

---

#### 2. Tối ưu mô hình bằng Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
```

**Giải thích:**

*   `GridSearchCV`: Lớp để tìm kiếm các tham số tốt nhất cho mô hình bằng cách thử tất cả các tổ hợp tham số có thể.
*   `param_grid`: Từ điển chứa các tham số và các giá trị cần thử.
*   `cv`: Số lượng fold trong cross-validation.
*   `scoring`: Hàm đánh giá hiệu suất của mô hình.
*   `best_params_`: Thuộc tính trả về các tham số tốt nhất.
*   `best_score_`: Thuộc tính trả về điểm số tốt nhất.

**📚 Tham khảo:**

*   [Scikit-learn documentation - GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

---

#### 3. Cross-validation (CV)

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring="accuracy")
print("CV Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())
```

**Giải thích:**

*   `cross_val_score`: Hàm để đánh giá hiệu suất của mô hình bằng cross-validation.
*   `cv`: Số lượng fold trong cross-validation.
*   `scoring`: Hàm đánh giá hiệu suất của mô hình.

**📚 Tham khảo:**

*   [Scikit-learn documentation - cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

---

#### 4. So sánh nhiều mô hình với Pipeline + GridSearch

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

param_grid = {
    "clf__C": [0.1, 1, 10],
    "clf__penalty": ["l2"],
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
```

**Giải thích:**

*   `Pipeline`: Cho phép bạn kết hợp nhiều bước xử lý dữ liệu và huấn luyện mô hình thành một quy trình duy nhất.
*   `StandardScaler`: Chuẩn hóa dữ liệu bằng cách loại bỏ giá trị trung bình và chia cho độ lệch chuẩn.
*   `clf__C`: Tham số C của mô hình Logistic Regression.
*   `clf__penalty`: Tham số penalty của mô hình Logistic Regression.

**📚 Tham khảo:**

*   [Scikit-learn documentation - Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
*   [Scikit-learn documentation - StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

### 🧪 Bài Lab Day 5

1.  Áp dụng `SelectKBest` để chọn 5 features tốt nhất
2.  Dùng RandomForest để hiển thị feature importance
3.  Dùng `GridSearchCV` tối ưu Logistic và RandomForest
4.  Ghi kết quả `best_params_` và `best_score_` vào file `model_result.json`

**Hướng dẫn:**

*   Sử dụng `SelectKBest` để chọn 5 đặc trưng tốt nhất từ tập dữ liệu Titanic.
*   Sử dụng `RandomForestClassifier` để huấn luyện mô hình và hiển thị độ quan trọng của các đặc trưng.
*   Sử dụng `GridSearchCV` để tìm các tham số tốt nhất cho mô hình `LogisticRegression` và `RandomForestClassifier`.
*   Lưu các tham số tốt nhất và điểm số vào file `model_result.json`.

---

### 📝 Bài tập về nhà Day 5:

1.  So sánh mô hình Logistic Regression, Decision Tree và Random Forest sau khi tuning
2.  Tạo file `compare_models.py` để hiển thị bảng so sánh accuracy, precision, recall
3.  Lưu các mô hình tốt nhất vào `models/`
4.  Ghi nhật ký tuning trong `notebooks/05_model_tuning.ipynb`
5.  Commit và đẩy lên GitHub

**Hướng dẫn:**

*   Sử dụng các mô hình `LogisticRegression`, `DecisionTreeClassifier`, và `RandomForestClassifier` đã được tối ưu hóa bằng `GridSearchCV`.
*   Tạo một bảng so sánh các chỉ số đánh giá hiệu suất của các mô hình (accuracy, precision, recall, f1-score).
*   Lưu các mô hình tốt nhất vào thư mục `models/`.
*   Ghi lại quá trình tuning và kết quả vào notebook `notebooks/05_model_tuning.ipynb`.
