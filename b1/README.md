### ✅ **Day 1 – Rapid Python + Git Refresher + Setup môi trường**

**🎯 Mục tiêu:** Làm chủ công cụ, môi trường phát triển và kiến thức Python nền tảng cho ML. Đảm bảo bạn có một workspace ổn định, hiện đại, hỗ trợ phát triển dự án DS/ML lâu dài.

**🏋️ Mục tiêu chung:**

* Thiết lập môi trường làm việc với Python cho ML
* Ôn tập nhanh Python (comprehension, OOP, typing, lambda, decorators, generators...)
* Có bài lab thực hành cuối mỗi phần

---

#### 1. Cài đặt môi trường & IDE

**☑️ Yêu cầu:**

*   **Python >= 3.9 (ưu tiên 3.10):**  Ngôn ngữ lập trình phổ biến nhất trong Machine Learning và AI.
*   **Công cụ quản lý môi trường:** `conda`, `venv`, hoặc `poetry`.  Giúp quản lý các môi trường Python riêng biệt cho từng dự án.
    *   **Conda:** Trình quản lý gói, quản lý môi trường, phụ thuộc đa nền tảng mã nguồn mở.
    *   **venv:** Mô-đun trong thư viện chuẩn Python để tạo môi trường ảo.
    *   **Poetry:** Công cụ quản lý sự phụ thuộc và đóng gói Python.
*   **IDE:** VS Code + Extension: Python, Jupyter, GitLens.

**✅ Hướng dẫn:**

1.  Cài Python >= 3.10 (dùng pyenv hoặc python.org)
2.  Cài VS Code + Extension: Python, Jupyter, GitLens
3.  Tạo cấu trúc dự án:

    ```bash
    mkdir my-ml-project && cd my-ml-project
    mkdir data models notebooks scripts src
    code .
    ```

4.  Cài môi trường (ví dụ với Conda):

    ```bash
    conda create -n ml_env python=3.10 -y
    conda activate ml_env
    pip install jupyterlab matplotlib numpy pandas
    ```

5.  Kiểm tra:

    ```bash
    python -c "import numpy; import pandas; print('OK')"
    ```

**📚 Tham khảo:**

*   [Conda documentation](https://docs.conda.io/en/latest/)
*   [venv documentation](https://docs.python.org/3/library/venv.html)
*   [Poetry documentation](https://python-poetry.org/docs/)

---

#### 2. Python Refresher – Lập trình hiện đại và tối ưu cho ML

##### **🧠 Kiến thức cần ôn & Ví dụ:**

*   **List / Dict comprehension:** Cú pháp ngắn gọn để tạo list và dict. Cho phép tạo list và dict một cách ngắn gọn và dễ đọc hơn.

    ```python
    # List comprehension: Tạo list bình phương các số chẵn từ 0-9
    squares = [x**2 for x in range(10) if x % 2 == 0]

    primes = [x for x in range(2, 100) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    text = "machine learning is fun"
    letter_freq = {c: text.count(c) for c in set(text) if c.isalpha()}
    ```

*   **Hàm lambda, map, filter:** Các công cụ mạnh mẽ để xử lý dữ liệu linh hoạt.
    *   **lambda:** Tạo hàm ẩn danh, thường dùng cho các hàm đơn giản.
    *   **map:** Áp dụng một hàm cho từng phần tử trong iterable.
    *   **filter:** Lọc các phần tử trong iterable dựa trên một điều kiện.

    ```python
    nums = list(range(10))
    double = list(map(lambda x: x * 2, nums))  # Nhân đôi các số
    odds = list(filter(lambda x: x % 2 == 1, nums)) # Lọc số lẻ
    from functools import reduce
    total = reduce(lambda x, y: x + y, nums) # Tính tổng các số
    ```

*   **Decorators:** Sửa đổi/mở rộng chức năng của hàm mà không cần thay đổi code gốc.  Là một hàm nhận một hàm khác làm đối số và trả về một hàm mới.

    ```python
    def logger(func):
        def wrapper(*args, **kwargs):
            print(f"[LOG] Running {func.__name__}")
            return func(*args, **kwargs)
        return wrapper

    @logger
    def greet(name):
        print(f"Hello, {name}")
    ```

*   **Generators & `yield`:** Tạo chuỗi giá trị tuần tự, tiết kiệm bộ nhớ.  Thay vì trả về một list lớn, generator trả về một iterator, giúp tiết kiệm bộ nhớ.

    ```python
    def fibonacci(n):
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b
    ```

*   **Context Manager (`with`):** Đảm bảo tài nguyên được giải phóng đúng cách.  Đảm bảo các tài nguyên như file, kết nối mạng được giải phóng, ngay cả khi có lỗi.

    ```python
    with open("example.txt", "w") as f:
        f.write("Hello context manager!")

    from contextlib import contextmanager
    @contextmanager
    def open_file(name, mode):
        f = open(name, mode)
        try:
            yield f
        finally:
            f.close()
    ```

*   **OOP nâng cao (kế thừa, `@dataclass`, `typing`, `__str__`, `__repr__`):** Xây dựng ứng dụng phức tạp dễ dàng hơn.
    *   **Kế thừa:** Cho phép một class kế thừa các thuộc tính và phương thức từ một class khác.
    *   `@dataclass`:  Decorator giúp tự động tạo các phương thức `__init__`, `__repr__`, `__eq__`.
    *   `typing`: Module cung cấp các type hint để chỉ định kiểu dữ liệu.
    *   `__str__`: Phương thức trả về chuỗi đại diện cho đối tượng (dễ đọc).
    *   `__repr__`: Phương thức trả về chuỗi đại diện cho đối tượng (cho debug).

    ```python
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class Student:
        name: str
        scores: List[int]

        def average(self) -> float:
            return sum(self.scores) / len(self.scores)

    class Animal:
        def speak(self):
            print("...")

    class Dog(Animal):
        def speak(self):
            print("Woof!")
    ```

**📚 Tham khảo:**

*   [Python List Comprehension](https://realpython.com/list-comprehension-python/)
*   [Python Lambda Functions](https://realpython.com/python-lambda/)
*   [Python Decorators](https://realpython.com/primer-on-python-decorators/)
*   [Python Generators](https://realpython.com/python-generators/)
*   [Python Context Managers](https://realpython.com/python-with-statement/)
*   [Python Dataclasses](https://realpython.com/python-data-classes/)
*   [Python OOP](https://realpython.com/python3-object-oriented-programming/)
*   [Python Type Hint](https://realpython.com/python-type-checking/)
*   [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/)
*   [PEP 526 -- Syntax for Variable Annotations](https://peps.python.org/pep-0526/)
*   [PEP 557 -- Data Classes](https://peps.python.org/pep-0557/)

---

##### **🧪 Lab thực hành:**

*   **List/Dict Comprehension:**
    *   Tạo ma trận 5x5 số ngẫu nhiên 0–9
    *   Dict: i -> i\*i với i chẳn, từ 1–20
*   **Lambda, map, filter:**
    *   map: căn bậc hai list số
    *   filter: từ dài > 5
    *   reduce: giai thừa 5
*   **Decorators:**
    *   `@timer`: đo thời gian chạy
    *   `@debug`: in đối số + kết quả
*   **Generators:**
    *   Tạo generator tổng dãy số chẳn
    *   Đọc dòng từ file lớn
*   **Context Manager:**
    *   Context log thời gian khởi chạy/kết thúc
*   **OOP + Dataclass + Typing:**
    *   Class `Person` với `introduce()`
    *   Kế thừa `Student` -> gpa
    *   Class `Product` dùng dataclass + `__str__`

---

#### 3. Git & GitHub Workflow chuyên nghiệp

**✅ Kiến thức chính:**

*   **Git init, branch, commit, push, pull, merge:** Các lệnh cơ bản để quản lý code bằng Git.
*   **PR (pull request), review, squash merge:** Quy trình làm việc nhóm hiệu quả trên GitHub.
*   **Viết README chuẩn, dùng GitHub Actions CI:** Các best practices để tạo một dự án chuyên nghiệp trên GitHub.

**📁 Cấu trúc thư mục chuẩn:**

```
my-ml-project/
├── data/               # Dữ liệu gốc, dữ liệu đã xử lý
├── models/             # File .pkl hoặc .joblib model đã train
├── notebooks/          # Phân tích dữ liệu, EDA, thử nghiệm mô hình
├── scripts/            # Các script hỗ trợ, train model, inference
├── src/                # Code logic chính
├── README.md           # Giới thiệu dự án
├── requirements.txt    # Thư viện cần cài
└── .gitignore
```

**📚 Tham khảo:**

*   [Git documentation](https://git-scm.com/doc)
*   [GitHub documentation](https://docs.github.com/)
*   [GitHub Actions documentation](https://docs.github.com/en/actions)

---

#### 4. Thiết lập pre-commit để tự động format code

**Giải thích:**

`pre-commit` là framework giúp quản lý các hook trước khi commit code, tự động format code, kiểm tra lỗi chính tả,...

**Hướng dẫn:**

1.  Cài đặt `pre-commit`:

    ```bash
    pip install pre-commit black flake8 isort
    pre-commit install
    ```

2.  Tạo file `.pre-commit-config.yaml`:

    ```yaml
    repos:
      - repo: https://github.com/psf/black
        rev: stable
        hooks:
          - id: black

      - repo: https://github.com/pycqa/flake8
        rev: 6.1.0
        hooks:
          - id: flake8

      - repo: https://github.com/timothycrosley/isort
        rev: 5.12.0
        hooks:
          - id: isort
    ```

**📚 Tham khảo:**

*   [pre-commit documentation](https://pre-commit.com/)
*   [black documentation](https://github.com/psf/black)
*   [flake8 documentation](https://flake8.pycqa.org/en/latest/)
*   [isort documentation](https://pycqa.github.io/isort/)

---

#### 🌟 Mini Project Cuối Buổi

Tạo `StudentManager`:

*   Thêm/Xóa/Hiển thị sinh viên
*   Class Student gồm: name, age, scores
*   Dùng comprehension + lambda + sort -> top 3 GPA
*   Dùng `@dataclass`, `typing.List`

---

#### 🧪 Bài Lab Day 1

**Hướng dẫn:**

Thực hành các kiến thức đã học.

1.  **Cài Conda + VS Code + JupyterLab + Tạo folder cấu trúc project ML:** Làm theo hướng dẫn ở phần 1.
2.  **Tạo file `main.py` và viết class `MLPipeline` sử dụng `@dataclass`:** Trong thư mục `src`.
3.  **Viết thử decorator + generator demo trong `src/utils.py`:**
4.  **Tạo `.git` repo, commit, push lên GitHub.**
5.  **Tích hợp `pre-commit` để format code tự động:** Làm theo hướng dẫn ở phần 4.

---

#### 📝 Bài tập về nhà Day 1:

1.  **Viết 3 đoạn code nhỏ:**

    *   Dùng list comprehension lọc và xử lý danh sách số nguyên.
    *   Một hàm có dùng `@decorator` tự viết.
    *   Một generator yield ra 5 số lẻ đầu tiên lớn hơn 100.

2.  **Đọc thêm về các chủ đề: `@dataclass`, `with` context manager, `typing`.**

3.  **Cấu hình VS Code format code tự động khi save file.**

4.  **Commit thay đổi bài tập vào repo GitHub đã tạo.**
