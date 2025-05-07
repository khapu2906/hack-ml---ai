### ✅ **Day 3 – Khám phá & làm sạch dữ liệu nâng cao (EDA)**


**🎯 Mục tiêu:** Nắm vững quy trình khám phá dữ liệu (EDA), phát hiện lỗi, hiểu rõ cấu trúc dữ liệu và làm sạch dữ liệu chuyên nghiệp trước khi train model.

---

#### 1. Tổng quan về EDA – Khám phá dữ liệu hiệu quả

**✅ Công việc chính:**

*   **Tổng quan: shape, missing, types:**
    *   **Giải thích:** Bước đầu tiên trong EDA là xem xét tổng quan về dữ liệu, bao gồm số lượng hàng và cột (`shape`), số lượng giá trị bị thiếu (`missing`), và kiểu dữ liệu của các cột (`types`).
*   **Mô tả thống kê: `.describe()`, `.value_counts()`, `.nunique()`:**
    *   **Giải thích:** Các hàm này cung cấp các thống kê mô tả về dữ liệu, bao gồm trung bình, độ lệch chuẩn, giá trị lớn nhất, giá trị nhỏ nhất, ...
*   **Kiểm tra phân phối, outliers, correlation:**
    *   **Giải thích:** Bước này giúp bạn hiểu rõ hơn về cách dữ liệu được phân bố, phát hiện các giá trị ngoại lệ và tìm hiểu mối quan hệ giữa các biến.
*   **Nhìn dữ liệu từ góc độ mục tiêu (`target`) → label leakage?**
    *   **Giải thích:** Xem xét dữ liệu từ góc độ biến mục tiêu giúp bạn xác định các biến có liên quan đến biến mục tiêu và phát hiện các trường hợp label leakage (rò rỉ thông tin từ biến mục tiêu vào các biến đầu vào).

---

---

#### 2. Xử lý dữ liệu thiếu (missing values)

**✅ Các kỹ thuật:**

*   **Xem tổng quan: `df.isna().sum()`, `missingno.matrix(df)`:**
    *   **Giải thích:** Các hàm này giúp bạn xem tổng quan về số lượng giá trị bị thiếu trong mỗi cột của DataFrame. `missingno.matrix(df)` tạo ra một biểu đồ trực quan cho thấy vị trí của các giá trị bị thiếu.
*   **Loại bỏ: `.dropna()`:**
    *   **Giải thích:** Hàm này cho phép bạn loại bỏ các hàng hoặc cột chứa giá trị bị thiếu.
*   **Điền giá trị phù hợp:**
        *   **Trung bình, trung vị: `.fillna(df.mean())`:** Điền các giá trị bị thiếu bằng giá trị trung bình hoặc trung vị của cột.
        *   **Forward/Backward fill: `.fillna(method='ffill')`:** Điền các giá trị bị thiếu bằng giá trị trước đó hoặc giá trị tiếp theo.
        *   **Tạo biến cờ `is_missing`:** Tạo một cột mới cho biết liệu một giá trị có bị thiếu hay không.

**🧪 Ví dụ:**

```python
df["Age_missing"] = df["Age"].isna().astype(int)
df["Age"] = df["Age"].fillna(df["Age"].median())
```

**📚 Tham khảo:**

*   [Pandas documentation - Handling missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
*   [Missingno documentation](https://github.com/ResidentMario/missingno)

---

#### 3. Phân tích outlier & phân phối

**✅ Công cụ:**

*   `boxplot`, `histplot`, `distplot`, `describe`:
    *   **Giải thích:** Các công cụ này giúp bạn trực quan hóa và mô tả phân phối của dữ liệu, từ đó phát hiện các giá trị ngoại lệ (outlier).
*   **Xác định ngưỡng: IQR (Q1–Q3), Z-score:**
    *   **Giải thích:** Các phương pháp này giúp bạn xác định ngưỡng để phân loại các giá trị là outlier.
        *   **IQR (Interquartile Range):** Khoảng giữa Q1 (quartile 25%) và Q3 (quartile 75%).
        *   **Z-score:** Số lượng độ lệch chuẩn mà một giá trị cách xa giá trị trung bình.

**🧪 Ví dụ:**

```python
import seaborn as sns
import numpy as np

# Boxplot phát hiện outlier
sns.boxplot(x=df["Fare"])

# Xử lý outlier: clip theo IQR
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
df["Fare_clipped"] = df["Fare"].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
```

**📚 Tham khảo:**

*   [Seaborn documentation](https://seaborn.pydata.org/)
*   [Box plot statistics](https://towardsdatascience.com/understanding-boxplots-5f2df72410bc)

---

#### 4. Phân tích tương quan & ma trận correlation

**✅ Kỹ thuật:**

*   `.corr()` để tính hệ số Pearson
    *   **Giải thích:** Hàm `corr()` tính hệ số tương quan Pearson giữa các cặp biến trong DataFrame. Hệ số này có giá trị từ -1 đến 1, cho biết mức độ và hướng của mối quan hệ tuyến tính giữa hai biến.
*   `sns.heatmap()` để trực quan hoá
    *   **Giải thích:** Hàm `heatmap()` từ thư viện Seaborn giúp bạn trực quan hóa ma trận tương quan bằng cách sử dụng màu sắc để biểu thị giá trị của các hệ số tương quan.

**🧪 Ví dụ:**

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/titanic.csv")
df = df.select_dtypes(include=['number'])
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

**📚 Tham khảo:**

*   [Pandas documentation - corr()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
*   [Seaborn documentation - heatmap()](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

---

#### 5. Label Encoding & Feature Engineering cơ bản

*   **Xử lý biến dạng text (categorical):**
    *   **`LabelEncoder`, `OneHotEncoder`:** Các công cụ này giúp bạn chuyển đổi các biến hạng mục (categorical) thành dạng số để có thể sử dụng trong các mô hình Machine Learning.
    *   **`.map()`, `.replace()`:** Các hàm này cho phép bạn thay thế các giá trị trong một cột bằng các giá trị khác.
*   **Tạo biến mới từ cột ngày tháng, tên, nhóm tuổi,…**
    *   **Giải thích:** Feature engineering là quá trình tạo ra các biến mới từ các biến hiện có để cải thiện hiệu suất của mô hình.

**🧪 Ví dụ:**

```python
import pandas as pd
df = pd.read_csv("data/titanic.csv")

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
```

**📚 Tham khảo:**

*   [Scikit-learn documentation - LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
*   [Scikit-learn documentation - OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
*   [Feature Engineering for Machine Learning](https://www.amazon.com/Feature-Engineering-Machine-Learning-Principles/dp/1491953247)

---

### 🧪 Bài Lab Day 3

**Hướng dẫn:**

Trong bài lab này, bạn sẽ thực hành các kiến thức đã học trong ngày 3 bằng cách thực hiện các bước sau:

1.  **Dùng Pandas & Seaborn để khám phá kỹ Titanic dataset:** Sử dụng các hàm của Pandas và Seaborn để khám phá các đặc trưng của bộ dữ liệu Titanic, bao gồm phân phối, outliers, và mối quan hệ giữa các biến.
2.  **Xử lý các cột có missing value (`Age`, `Cabin`, `Embarked`):** Áp dụng các kỹ thuật xử lý missing values đã học để điền hoặc loại bỏ các giá trị bị thiếu trong các cột `Age`, `Cabin`, và `Embarked`.
3.  **Tạo thêm biến: `AgeGroup`, `FareGroup`, `HasCabin`, `Title`:** Sử dụng các kỹ thuật feature engineering để tạo ra các biến mới từ các biến hiện có.
4.  **Phân tích tương quan giữa các biến với `Survived`:** Sử dụng các hàm `corr()` và `heatmap()` để phân tích mối tương quan giữa các biến và biến mục tiêu `Survived`.

---

### 📝 Bài tập về nhà Day 3:

1.  **Viết hàm Python `detect_outliers(df, column)` dùng IQR hoặc Z-score để in ra index các dòng nghi ngờ là outlier.**
    *   **Gợi ý:** Hàm này nên nhận một DataFrame và tên cột làm đầu vào, sau đó sử dụng IQR hoặc Z-score để xác định các outlier và trả về danh sách các index của các dòng chứa outlier.
2.  **Lưu `titanic_cleaned.csv` vào thư mục `data/processed/` sau khi làm sạch:**
    *   **Gợi ý:** Sử dụng hàm `to_csv()` của Pandas để lưu DataFrame vào file CSV. Đảm bảo tạo thư mục `data/processed/` trước nếu nó chưa tồn tại.
3.  **Ghi toàn bộ phân tích + hình ảnh vào notebook `titanic_eda.ipynb`:**
    *   **Gợi ý:** Sử dụng Markdown để ghi chú rõ ràng các bước thực hiện và kết quả phân tích. Chèn các hình ảnh biểu đồ vào notebook để trực quan hóa dữ liệu.
4.  **Commit + push code + notebook lên GitHub:** Commit các thay đổi vào Git và đẩy lên GitHub.

---
