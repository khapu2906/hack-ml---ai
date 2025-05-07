### ✅ **Day 4 – Huấn luyện mô hình ML đầu tiên với Scikit-learn**


**🎯 Mục tiêu:** Hiểu quy trình Machine Learning cơ bản (pipeline), thực hành xây dựng mô hình phân loại đầu tiên bằng `Scikit-learn`.

---

#### 1. Giới thiệu quy trình ML chuẩn (Sklearn Pipeline)

**Quy trình tổng quát:**

```
Data → Clean → Feature Engineering → Train/Test split → Train model → Evaluate → Save model
```

**✅ Thư viện chính:**

*   `scikit-learn`: model, pipeline, metrics
*   `joblib`: lưu mô hình
*   `sklearn.model_selection`: `train_test_split`, `cross_val_score`

**Giải thích:**

*   **Data**: Thu thập dữ liệu từ các nguồn khác nhau.
*   **Clean**: Làm sạch dữ liệu bằng cách xử lý các giá trị bị thiếu, loại bỏ các outlier, ...
*   **Feature Engineering**: Tạo ra các biến mới từ các biến hiện có để cải thiện hiệu suất của mô hình.
*   **Train/Test split**: Chia dữ liệu thành hai tập: tập huấn luyện và tập kiểm tra.
*   **Train model**: Huấn luyện mô hình trên tập huấn luyện.
*   **Evaluate**: Đánh giá hiệu suất của mô hình trên tập kiểm tra.
*   **Save model**: Lưu mô hình đã huấn luyện để sử dụng sau này.

**📚 Tham khảo:**

*   [Scikit-learn documentation](https://scikit-learn.org/stable/index.html)
*   [Joblib documentation](https://joblib.readthedocs.io/en/latest/)

---

#### 2. Train/Test Split + Xác định biến đầu vào & mục tiêu

```python
from sklearn.model_selection import train_test_split

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Giải thích:**

*   `X`: Ma trận các biến đầu vào.
*   `y`: Vector biến mục tiêu.
*   `test_size`: Tỉ lệ kích thước của tập kiểm tra so với tập dữ liệu ban đầu.
*   `random_state`: Seed cho bộ sinh số ngẫu nhiên.
*   `stratify`: Chia dữ liệu sao cho tỉ lệ các lớp trong tập huấn luyện và tập kiểm tra là tương đương.

**📚 Tham khảo:**

*   [Scikit-learn documentation - train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

#### 3. Huấn luyện mô hình Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```

**Giải thích:**

*   `LogisticRegression`: Mô hình hồi quy logistic.
*   `fit()`: Huấn luyện mô hình trên tập huấn luyện.
*   `predict()`: Dự đoán nhãn cho tập kiểm tra.
*   `classification_report()`: In ra các chỉ số đánh giá hiệu suất của mô hình, bao gồm precision, recall, f1-score, ...
*   `accuracy_score()`: Tính độ chính xác của mô hình.

**📚 Tham khảo:**

*   [Scikit-learn documentation - LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
*   [Scikit-learn documentation - classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
*   [Scikit-learn documentation - accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

---

#### 4. Đánh giá mô hình (metrics & confusion matrix)

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

**Giải thích:**

*   `confusion_matrix()`: Tính ma trận nhầm lẫn.
*   `sns.heatmap()`: Vẽ ma trận nhầm lẫn bằng thư viện Seaborn.

**📚 Tham khảo:**

*   [Scikit-learn documentation - confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
*   [Seaborn documentation - heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

---

#### 5. Lưu mô hình đã huấn luyện & tái sử dụng

```python
import joblib

# Lưu mô hình
joblib.dump(model, "models/logistic_model.pkl")

# Load lại mô hình
loaded_model = joblib.load("models/logistic_model.pkl")
```

**Giải thích:**

*   `joblib.dump()`: Lưu mô hình đã huấn luyện vào file.
*   `joblib.load()`: Tải mô hình đã lưu từ file.

**📚 Tham khảo:**

*   [Joblib documentation](https://joblib.readthedocs.io/en/latest/)

---

#### 6. Bonus: Pipeline Scikit-learn chuyên nghiệp

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

**Giải thích:**

*   `Pipeline`: Cho phép bạn kết hợp nhiều bước xử lý dữ liệu và huấn luyện mô hình thành một quy trình duy nhất.
*   `StandardScaler`: Chuẩn hóa dữ liệu bằng cách loại bỏ giá trị trung bình và chia cho độ lệch chuẩn.

**📚 Tham khảo:**

*   [Scikit-learn documentation - Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
*   [Scikit-learn documentation - StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

### 🧪 Bài Lab Day 4

1.  Tải Titanic dataset đã làm sạch (`data/processed/titanic_cleaned.csv`)
2.  Train/test split và xây Logistic Regression
3.  Tính accuracy + classification report + confusion matrix
4.  Save mô hình `.pkl` vào thư mục `models/`

**Hướng dẫn:**

*   Sử dụng Pandas để tải dữ liệu đã làm sạch.
*   Sử dụng `train_test_split` để chia dữ liệu thành tập huấn luyện và tập kiểm tra.
*   Sử dụng `LogisticRegression` để huấn luyện mô hình.
*   Sử dụng `accuracy_score`, `classification_report`, và `confusion_matrix` để đánh giá mô hình.
*   Sử dụng `joblib.dump` để lưu mô hình.

---

### 📝 Bài tập về nhà Day 4:

1.  Thử thay Logistic Regression bằng:
    *   Decision Tree (`DecisionTreeClassifier`)
    *   Random Forest (`RandomForestClassifier`)
2.  So sánh kết quả giữa các mô hình
3.  Viết lại quá trình train và đánh giá trong file `train_model.py`
4.  Commit toàn bộ code + mô hình lên GitHub

**Hướng dẫn:**

*   Sử dụng `DecisionTreeClassifier` và `RandomForestClassifier` để huấn luyện các mô hình khác.
*   Sử dụng các chỉ số đánh giá để so sánh hiệu suất của các mô hình.
*   Viết code vào file `train_model.py` để có thể chạy lại quá trình huấn luyện và đánh giá mô hình một cách dễ dàng.
