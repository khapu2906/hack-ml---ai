### ✅ **Day 2 – Phân tích dữ liệu với NumPy & Pandas**


**🎯 Mục tiêu:** Làm chủ thao tác dữ liệu bằng NumPy & Pandas – công cụ quan trọng nhất trong giai đoạn phân tích và tiền xử lý dữ liệu.

---

#### 1. Làm quen với NumPy – "trái tim" tính toán của ML

**✅ Kiến thức chính:**

*   **Tạo mảng `ndarray`, reshape, indexing, slicing:**
    *   **Giải thích:** `ndarray` là kiểu dữ liệu mảng đa chiều cơ bản trong NumPy. `reshape` cho phép thay đổi hình dạng của mảng. `indexing` và `slicing` cho phép truy cập và trích xuất các phần tử của mảng.
    *   **Ví dụ:**
        ```python
        import numpy as np
        a = np.array([1, 2, 3, 4, 5, 6])
        b = a.reshape((2, 3))  # Reshape thành mảng 2x3
        print(b[0, 1])  # Truy cập phần tử ở hàng 0, cột 1
        print(a[1:4])  # Slicing từ phần tử 1 đến 3
        ```
*   **Broadcasting và vectorized operations:**
    *   **Giải thích:** Broadcasting cho phép thực hiện các phép toán trên các mảng có hình dạng khác nhau. Vectorized operations cho phép thực hiện các phép toán trên toàn bộ mảng một cách nhanh chóng và hiệu quả.
    *   **Ví dụ:**
        ```python
        import numpy as np
        a = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2])
        print(a + b)  # Broadcasting: b được tự động mở rộng để có cùng shape với a
        ```
*   **Hàm thống kê: `mean`, `std`, `argmax`, `sum`, `axis=`:**
    *   **Giải thích:** NumPy cung cấp các hàm thống kê để tính toán các giá trị như trung bình, độ lệch chuẩn, giá trị lớn nhất, tổng, ... Tham số `axis` cho phép bạn chỉ định trục mà bạn muốn tính toán trên đó.
    *   **Ví dụ:**
        ```python
        import numpy as np
        a = np.array([[1, 2], [3, 4]])
        print(a.mean(axis=0))  # Tính trung bình theo cột
        print(a.sum(axis=1))  # Tính tổng theo hàng
        ```
*   **Random: `np.random`, seed, tạo mẫu ngẫu nhiên:**
    *   **Giải thích:** NumPy cung cấp module `np.random` để tạo các số ngẫu nhiên. `seed` cho phép bạn khởi tạo bộ sinh số ngẫu nhiên để đảm bảo tính tái lặp.
    *   **Ví dụ:**
        ```python
        import numpy as np
        np.random.seed(42)  # Khởi tạo seed để đảm bảo tính tái lặp
        rand_data = np.random.normal(loc=0, scale=1, size=(3, 4))  # Tạo mảng ngẫu nhiên theo phân phối chuẩn
        ```

**🧪 Ví dụ:**

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a.mean(axis=0))  # Trung bình theo cột

# Broadcasting
b = np.array([1, 2])
print(a + b)  # Tự động mở rộng shape

# Tạo mảng ngẫu nhiên
np.random.seed(42)
rand_data = np.random.normal(loc=0, scale=1, size=(3, 4))
```

**📚 Tham khảo:**

*   [NumPy documentation](https://numpy.org/doc/stable/)
*   [NumPy tutorial](https://numpy.org/doc/stable/user/quickstart.html)

---

#### 2. Làm chủ Pandas – "xương sống" phân tích dữ liệu

**✅ Kiến thức chính:**

*   **Series vs DataFrame:**
    *   **Giải thích:** `Series` là một mảng một chiều có gắn nhãn, trong khi `DataFrame` là một bảng hai chiều có cấu trúc cột.
    *   **Ví dụ:**
        ```python
        import pandas as pd
        s = pd.Series([1, 3, 5, np.nan, 6, 8])  # Tạo Series
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})  # Tạo DataFrame
        ```
*   **Đọc & ghi CSV/Excel:**
    *   **Giải thích:** Pandas cung cấp các hàm để đọc dữ liệu từ các file CSV và Excel, cũng như ghi dữ liệu vào các file này.
    *   **Ví dụ:**
        ```python
        import pandas as pd
        df = pd.read_csv("data/titanic.csv")  # Đọc file CSV
        df.to_excel("data/titanic.xlsx")  # Ghi file Excel
        ```
*   `.info()`, `.describe()`, `.head()`, `.value_counts()`:
    *   **Giải thích:** Các hàm này cung cấp thông tin tổng quan về DataFrame, bao gồm kiểu dữ liệu của các cột, thống kê mô tả, hiển thị một vài dòng đầu tiên và đếm số lượng giá trị duy nhất trong một cột.
    *   **Ví dụ:**
        ```python
        import pandas as pd
        df = pd.read_csv("data/titanic.csv")
        print(df.info())  # In thông tin về DataFrame
        print(df.describe())  # In thống kê mô tả
        print(df.head())  # In 5 dòng đầu tiên
        print(df["Sex"].value_counts())  # Đếm số lượng giá trị duy nhất trong cột "Sex"
        ```
*   **Filtering, `.loc[]`, `.iloc[]`, mask:**
    *   **Giải thích:** Các công cụ này cho phép bạn lọc dữ liệu dựa trên các điều kiện, truy cập các phần tử của DataFrame theo nhãn hoặc vị trí, và tạo mask để chọn các phần tử thỏa mãn một điều kiện.
    *   **Ví dụ:**
        ```python
        import pandas as pd
        df = pd.read_csv("data/titanic.csv")
        female_passengers = df[df["Sex"] == "female"]  # Lọc hành khách nữ
        print(df.loc[0, "Name"])  # Truy cập phần tử ở hàng 0, cột "Name"
        print(df.iloc[0, 0])  # Truy cập phần tử ở hàng 0, cột 0
        mask = df["Age"] > 30  # Tạo mask cho những người trên 30 tuổi
        print(df[mask])  # Lọc DataFrame dựa trên mask
        ```

---

*   **Thêm/xoá/sửa cột:**
    *   **Giải thích:** Pandas cho phép bạn dễ dàng thêm, xoá và sửa đổi các cột trong DataFrame.
    *   **Ví dụ:**
        ```python
        import pandas as pd
        df = pd.read_csv("data/titanic.csv")
        df["FamilySize"] = df["SibSp"] + df["Parch"]  # Thêm cột mới
        df.drop("Cabin", axis=1, inplace=True)  # Xoá cột "Cabin"
        df["Age"] = df["Age"] + 1  # Sửa đổi cột "Age"
        ```
*   **Xử lý missing data: `.isna()`, `.fillna()`, `.dropna()`:**
    *   **Giải thích:** Pandas cung cấp các hàm để phát hiện và xử lý dữ liệu bị thiếu. `.isna()` trả về một DataFrame boolean cho biết các giá trị nào bị thiếu. `.fillna()` cho phép bạn điền các giá trị bị thiếu bằng một giá trị cụ thể. `.dropna()` cho phép bạn xoá các hàng hoặc cột chứa giá trị bị thiếu.
    *   **Ví dụ:**
        ```python
        import pandas as pd
        df = pd.read_csv("data/titanic.csv")
        print(df.isna().sum())  # Đếm số lượng giá trị bị thiếu trong mỗi cột
        df["Age"] = df["Age"].fillna(df["Age"].mean())  # Điền giá trị bị thiếu trong cột "Age" bằng giá trị trung bình
        df.dropna(inplace=True)  # Xoá các hàng chứa giá trị bị thiếu
        ```

**📚 Tham khảo:**

*   [Pandas documentation](https://pandas.pydata.org/docs/)
*   [Pandas tutorial](https://pandas.pydata.org/docs/user_guide/10min.html)
*   [Wes McKinney. Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference, 2010.](https://www.dlr.de/sc/en/desktopdefault.aspx/tabid-10628/15193_read-38661/year-all/)

---

#### 3. Visualization cơ bản bằng Pandas + Matplotlib

**✅ Kiến thức chính:**

*   **Histogram, boxplot, barplot:**
    *   **Giải thích:** Các biểu đồ này là các công cụ hữu ích để trực quan hóa dữ liệu và khám phá các mối quan hệ giữa các biến.
        *   **Histogram:** Biểu đồ tần suất, hiển thị phân phối của một biến liên tục.
        *   **Boxplot:** Biểu đồ hộp, hiển thị các giá trị tứ phân vị, giá trị trung bình và các giá trị ngoại lệ của một biến liên tục.
        *   **Barplot:** Biểu đồ cột, hiển thị giá trị trung bình của một biến theo các nhóm khác nhau.
    *   **Ví dụ:**
        ```python
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv("data/titanic.csv")

        df["Age"].hist(bins=20)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.show()

        df.boxplot(column="Age", by="Sex")
        plt.show()

        df.groupby("Sex")["Survived"].mean().plot(kind="bar")
        plt.show()
        ```
*   **Phân tích phân phối, phát hiện outlier:**
    *   **Giải thích:** Phân tích phân phối giúp bạn hiểu rõ hơn về cách dữ liệu được phân bố. Phát hiện outlier giúp bạn xác định các giá trị bất thường có thể ảnh hưởng đến kết quả phân tích.

**🧪 Ví dụ:**

```python
import matplotlib.pyplot as plt

df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

**📚 Tham khảo:**

*   [Matplotlib documentation](https://matplotlib.org/stable/contents.html)
*   [Seaborn documentation](https://seaborn.pydata.org/)

---

*   **Phân tích phân phối, phát hiện outlier:**
    *   **Giải thích:** Phân tích phân phối giúp bạn hiểu rõ hơn về cách dữ liệu được phân bố. Phát hiện outlier giúp bạn xác định các giá trị bất thường có thể ảnh hưởng đến kết quả phân tích.

**🧪 Ví dụ:**

```python
import matplotlib.pyplot as plt

df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

**📚 Tham khảo:**

*   [Matplotlib documentation](https://matplotlib.org/stable/contents.html)
*   [Seaborn documentation](https://seaborn.pydata.org/)

---

#### 4. Phân tích dữ liệu thực tế: Titanic Dataset

*   **Download:** [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
*   **Đặt file `train.csv` vào thư mục `data/`**

**Giải thích:**

Trong phần này, chúng ta sẽ thực hành các kiến thức đã học bằng cách phân tích bộ dữ liệu Titanic. Bộ dữ liệu này chứa thông tin về các hành khách trên tàu Titanic, bao gồm thông tin về tuổi, giới tính, hạng vé, ... Mục tiêu là dự đoán xem hành khách nào có khả năng sống sót cao hơn.

**🧪 Một số câu hỏi gợi ý:**

*   Tỉ lệ sống sót (`Survived`) theo giới tính?
*   Độ tuổi trung bình của người sống sót vs không sống sót?
*   Bao nhiêu người có vé hạng nhất (`Pclass == 1`)?

**Gợi ý:**

Sử dụng các hàm của Pandas như `groupby()`, `mean()`, `value_counts()` để trả lời các câu hỏi trên.

**📚 Tham khảo:**

*   [Titanic dataset](https://www.kaggle.com/c/titanic/data)
*   [Pandas documentation](https://pandas.pydata.org/docs/)

---

*   **Phân tích phân phối, phát hiện outlier:**
    *   **Giải thích:** Phân tích phân phối giúp bạn hiểu rõ hơn về cách dữ liệu được phân bố. Phát hiện outlier giúp bạn xác định các giá trị bất thường có thể ảnh hưởng đến kết quả phân tích.

**🧪 Ví dụ:**

```python
import matplotlib.pyplot as plt

df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

**📚 Tham khảo:**

*   [Matplotlib documentation](https://matplotlib.org/stable/contents.html)
*   [Seaborn documentation](https://seaborn.pydata.org/)

---

#### 4. Phân tích dữ liệu thực tế: Titanic Dataset

*   **Download:** [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
*   **Đặt file `train.csv` vào thư mục `data/`**

**Giải thích:**

Trong phần này, chúng ta sẽ thực hành các kiến thức đã học bằng cách phân tích bộ dữ liệu Titanic. Bộ dữ liệu này chứa thông tin về các hành khách trên tàu Titanic, bao gồm thông tin về tuổi, giới tính, hạng vé, ... Mục tiêu là dự đoán xem hành khách nào có khả năng sống sót cao hơn.

**🧪 Một số câu hỏi gợi ý:**

*   Tỉ lệ sống sót (`Survived`) theo giới tính?
*   Độ tuổi trung bình của người sống sót vs không sống sót?
*   Bao nhiêu người có vé hạng nhất (`Pclass == 1`)?

**Gợi ý:**

Sử dụng các hàm của Pandas như `groupby()`, `mean()`, `value_counts()` để trả lời các câu hỏi trên.

**📚 Tham khảo:**

*   [Titanic dataset](https://www.kaggle.com/c/titanic/data)
*   [Pandas documentation](https://pandas.pydata.org/docs/)

---

#### 4. Phân tích dữ liệu thực tế: Titanic Dataset

*   **Download:** [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
*   **Đặt file `train.csv` vào thư mục `data/`**

**Giải thích:**

Trong phần này, chúng ta sẽ thực hành các kiến thức đã học bằng cách phân tích bộ dữ liệu Titanic. Bộ dữ liệu này chứa thông tin về các hành khách trên tàu Titanic, bao gồm thông tin về tuổi, giới tính, hạng vé, ... Mục tiêu là dự đoán xem hành khách nào có khả năng sống sót cao hơn.

**🧪 Một số câu hỏi gợi ý:**

*   Tỉ lệ sống sót (`Survived`) theo giới tính?
*   Độ tuổi trung bình của người sống sót vs không sống sót?
*   Bao nhiêu người có vé hạng nhất (`Pclass == 1`)?

**Gợi ý:**

Sử dụng các hàm của Pandas như `groupby()`, `mean()`, `value_counts()` để trả lời các câu hỏi trên.

**📚 Tham khảo:**

*   [Titanic dataset](https://www.kaggle.com/c/titanic/data)
*   [Pandas documentation](https://pandas.pydata.org/docs/)

---

### 🧪 Bài Lab Day 2

**Hướng dẫn:**

Trong bài lab này, bạn sẽ thực hành các kiến thức đã học trong ngày 2.

1.  **Tải Titanic dataset và load bằng Pandas:** Tải bộ dữ liệu Titanic từ Kaggle và đọc nó vào một DataFrame Pandas.
2.  **Làm sạch dữ liệu: xử lý missing values, tạo biến mới (`FamilySize`):** Xử lý các giá trị bị thiếu trong bộ dữ liệu và tạo một cột mới `FamilySize` bằng cách kết hợp số lượng anh chị em/vợ chồng và số lượng cha mẹ/con cái.
3.  **Phân tích tỉ lệ sống sót theo `Sex`, `Pclass`:** Sử dụng các hàm `groupby()` và `mean()` để tính tỉ lệ sống sót theo giới tính và hạng vé.
4.  **Vẽ biểu đồ phân phối tuổi (`Age`) và so sánh giữa sống vs chết:** Sử dụng Matplotlib để vẽ biểu đồ phân phối tuổi cho những người sống sót và những người không sống sót.

---

### 🧪 Bài Lab Day 2

**Hướng dẫn:**

Trong bài lab này, bạn sẽ thực hành các kiến thức đã học trong ngày 2.

1.  **Tải Titanic dataset và load bằng Pandas:** Tải bộ dữ liệu Titanic từ Kaggle và đọc nó vào một DataFrame Pandas.
2.  **Làm sạch dữ liệu: xử lý missing values, tạo biến mới (`FamilySize`):** Xử lý các giá trị bị thiếu trong bộ dữ liệu và tạo một cột mới `FamilySize` bằng cách kết hợp số lượng anh chị em/vợ chồng và số lượng cha mẹ/con cái.
3.  **Phân tích tỉ lệ sống sót theo `Sex`, `Pclass`:** Sử dụng các hàm `groupby()` và `mean()` để tính tỉ lệ sống sót theo giới tính và hạng vé.
4.  **Vẽ biểu đồ phân phối tuổi (`Age`) và so sánh giữa sống vs chết:** Sử dụng Matplotlib để vẽ biểu đồ phân phối tuổi cho những người sống sót và những người không sống sót.

---

### 🧪 Bài Lab Day 2

**Hướng dẫn:**

Trong bài lab này, bạn sẽ thực hành các kiến thức đã học trong ngày 2.

1.  **Tải Titanic dataset và load bằng Pandas:** Tải bộ dữ liệu Titanic từ Kaggle và đọc nó vào một DataFrame Pandas.
    *   **Gợi ý:** Sử dụng hàm `pd.read_csv()` để đọc file CSV.
2.  **Làm sạch dữ liệu: xử lý missing values, tạo biến mới (`FamilySize`):** Xử lý các giá trị bị thiếu trong bộ dữ liệu và tạo một cột mới `FamilySize` bằng cách kết hợp số lượng anh chị em/vợ chồng và số lượng cha mẹ/con cái.
    *   **Gợi ý:** Sử dụng các hàm `isna()`, `fillna()`, `dropna()` để xử lý missing values. Sử dụng phép toán cộng để tạo cột mới.
3.  **Phân tích tỉ lệ sống sót theo `Sex`, `Pclass`:** Sử dụng các hàm `groupby()` và `mean()` để tính tỉ lệ sống sót theo giới tính và hạng vé.
4.  **Vẽ biểu đồ phân phối tuổi (`Age`) và so sánh giữa sống vs chết:** Sử dụng Matplotlib để vẽ biểu đồ phân phối tuổi cho những người sống sót và những người không sống sót.
    *   **Gợi ý:** Sử dụng hàm `hist()` để vẽ biểu đồ phân phối.

---

### 🧪 Bài Lab Day 2

**Hướng dẫn:**

Trong bài lab này, bạn sẽ thực hành các kiến thức đã học trong ngày 2.

1.  **Tải Titanic dataset và load bằng Pandas:** Tải bộ dữ liệu Titanic từ Kaggle và đọc nó vào một DataFrame Pandas.
    *   **Gợi ý:** Sử dụng hàm `pd.read_csv()` để đọc file CSV.
2.  **Làm sạch dữ liệu: xử lý missing values, tạo biến mới (`FamilySize`):** Xử lý các giá trị bị thiếu trong bộ dữ liệu và tạo một cột mới `FamilySize` bằng cách kết hợp số lượng anh chị em/vợ chồng và số lượng cha mẹ/con cái.
    *   **Gợi ý:** Sử dụng các hàm `isna()`, `fillna()`, `dropna()` để xử lý missing values. Sử dụng phép toán cộng để tạo cột mới.
3.  **Phân tích tỉ lệ sống sót theo `Sex`, `Pclass`:** Sử dụng các hàm `groupby()` và `mean()` để tính tỉ lệ sống sót theo giới tính và hạng vé.
4.  **Vẽ biểu đồ phân phối tuổi (`Age`) và so sánh giữa sống vs chết:** Sử dụng Matplotlib để vẽ biểu đồ phân phối tuổi cho những người sống sót và những người không sống sót.
    *   **Gợi ý:** Sử dụng hàm `hist()` để vẽ biểu đồ phân phối.

---

### 📝 Bài tập về nhà Day 2:

1.  **Tìm hiểu `.groupby()` + `.agg()` và thực hiện:**
    *   **Tính tỉ lệ sống sót trung bình theo `Sex` và `Pclass`:** Sử dụng các hàm `groupby()` và `agg()` để tính tỉ lệ sống sót trung bình theo giới tính và hạng vé.
        *   **Gợi ý:** Sử dụng hàm `groupby()` để nhóm dữ liệu theo `Sex` và `Pclass`, sau đó sử dụng hàm `agg()` để tính trung bình của cột `Survived`.
    *   **Đếm số người đi theo từng `Embarked` (`.value_counts()`):** Sử dụng hàm `value_counts()` để đếm số lượng người đi theo từng cảng lên tàu.
        *   **Gợi ý:** Sử dụng hàm `value_counts()` trên cột `Embarked`.
2.  **Viết 1 notebook `notebooks/titanic_analysis.ipynb`:**
    *   **Tải data, làm sạch, phân tích, vẽ biểu đồ:** Tạo một notebook Jupyter và thực hiện các bước sau:
        *   Tải bộ dữ liệu Titanic từ Kaggle.
        *   Làm sạch dữ liệu bằng cách xử lý các giá trị bị thiếu.
        *   Phân tích dữ liệu bằng cách tính toán các thống kê mô tả và vẽ biểu đồ.
    *   **Ghi chú rõ ràng bằng Markdown:** Sử dụng Markdown để ghi chú rõ ràng các bước thực hiện và kết quả phân tích.
3.  **Commit và push notebook lên GitHub:** Commit notebook lên GitHub để lưu lại và chia sẻ với người khác.
4.  **Challenge (tùy chọn):**
    *   **Sử dụng `seaborn` vẽ biểu đồ `boxplot` cho cột `Age` theo `Survived`:** Sử dụng thư viện Seaborn để vẽ biểu đồ boxplot cho cột `Age` theo `Survived`.
        *   **Gợi ý:** Sử dụng hàm `sns.boxplot()` để vẽ biểu đồ boxplot.

