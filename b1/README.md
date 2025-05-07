
### ✅ **Day 1 – Rapid Python + Git Refresher + Setup môi trường**

**🎯 Mục tiêu:** Làm chủ công cụ, môi trường phát triển và kiến thức Python nền tảng cho ML. Đảm bảo bạn có một workspace ổn định, hiện đại, hỗ trợ phát triển dự án DS/ML lâu dài.

---

#### 1. Cài đặt môi trường

**☑️ Yêu cầu:**

*   **Python >= 3.9 (ưu tiên 3.10):** Python là ngôn ngữ lập trình phổ biến nhất trong lĩnh vực Machine Learning và AI. Phiên bản 3.8 trở lên cung cấp nhiều tính năng và cải tiến quan trọng cho việc phát triển các ứng dụng ML.
*   **Công cụ quản lý môi trường: `conda` hoặc `venv`, `poetry`:** Các công cụ này giúp bạn tạo và quản lý các môi trường Python riêng biệt cho từng dự án. Điều này giúp tránh xung đột giữa các thư viện và đảm bảo tính ổn định của dự án.
    *   **Conda:** Một trình quản lý gói, quản lý môi trường, phụ thuộc đa nền tảng mã nguồn mở.
    *   **venv:** Mô-đun venv là một phần của thư viện chuẩn Python để tạo môi trường ảo.
    *   **Poetry:** Một công cụ để quản lý sự phụ thuộc và đóng gói Python.
*   **IDE: VS Code + Extension: Python, Jupyter, GitLens:** VS Code là một IDE mạnh mẽ và miễn phí, được nhiều nhà phát triển ML ưa chuộng. Các extension Python, Jupyter và GitLens giúp bạn viết code Python, chạy notebook Jupyter và quản lý code Git một cách hiệu quả.

**✅ Hướng dẫn:**

Để bắt đầu, bạn cần cài đặt Python và một trong các công cụ quản lý môi trường (Conda, venv hoặc Poetry). Sau đó, bạn có thể tạo một môi trường ảo cho dự án của mình và cài đặt các thư viện cần thiết.

```bash
# Với Conda:
conda create -n ml_env python=3.10 -y
conda activate ml_env
pip install jupyterlab notebook

# Với venv:
python -m venv venv
source venv/bin/activate  # hoặc venv\Scripts\activate
pip install jupyterlab
```

Tạo project folder mẫu:

```bash
mkdir my-ml-project && cd my-ml-project
mkdir data models notebooks scripts src
```

Mở bằng VS Code: `code .`

**📚 Tham khảo:**

*   [Conda documentation](https://docs.conda.io/en/latest/)
*   [venv documentation](https://docs.python.org/3/library/venv.html)
*   [Poetry documentation](https://python-poetry.org/docs/)

---

#### 2. Python Refresher – Lập trình hiện đại và tối ưu cho ML

##### **🧠 Kiến thức cần ôn:**

*   **List / Dict comprehension:** Cú pháp ngắn gọn để tạo list và dict.
    *   **Giải thích:** List và dict comprehension là những cú pháp ngắn gọn cho phép bạn tạo list và dict một cách nhanh chóng và dễ đọc. Thay vì sử dụng vòng lặp `for` để tạo list và dict, bạn có thể sử dụng list và dict comprehension để viết code ngắn gọn hơn và dễ đọc hơn.
    *   **Ví dụ:** `squares = [x**2 for x in range(10)]` tạo một list chứa bình phương của các số từ 0 đến 9.
    *   **Cú pháp:** `[expression for item in iterable if condition]`
*   **Hàm lambda, map, filter:** Các công cụ mạnh mẽ để xử lý dữ liệu một cách linh hoạt.
    *   **Giải thích:** Hàm lambda là một hàm ẩn danh, có nghĩa là nó không có tên. Hàm lambda thường được sử dụng để tạo các hàm nhỏ và đơn giản. Hàm `map` áp dụng một hàm cho mỗi phần tử trong một iterable và trả về một iterable mới chứa các kết quả. Hàm `filter` lọc các phần tử trong một iterable dựa trên một điều kiện và trả về một iterable mới chứa các phần tử thỏa mãn điều kiện.
    *   **Ví dụ:** `map(lambda x: x*2, numbers)` nhân đôi mỗi phần tử trong list `numbers`.
    *   **Cú pháp:**
        *   `lambda arguments: expression`
        *   `map(function, iterable)`
        *   `filter(function, iterable)`
*   **Decorators (hàm bao):** Cách để sửa đổi hoặc mở rộng chức năng của một hàm mà không cần thay đổi code của hàm đó.
    *   **Giải thích:** Decorator là một hàm nhận một hàm khác làm đối số và trả về một hàm mới. Decorator thường được sử dụng để thêm chức năng vào một hàm mà không cần thay đổi code của hàm đó.
    *   **Ví dụ:** `@timer` ở dưới để đo thời gian chạy của hàm `slow_func`.
    *   **Cú pháp:**
        ```python
        @decorator_name
        def function_name(arguments):
            # function body
        ```
*   **Generators & `yield`:** Cho phép tạo ra một chuỗi các giá trị một cách tuần tự, thay vì tạo ra một list lớn trong bộ nhớ.
    *   **Giải thích:** Generator là một hàm đặc biệt tạo ra một chuỗi các giá trị bằng cách sử dụng từ khóa `yield`. Thay vì trả về một list lớn chứa tất cả các giá trị, generator trả về một iterator cho phép bạn truy cập các giá trị một cách tuần tự. Điều này giúp tiết kiệm bộ nhớ, đặc biệt khi bạn cần xử lý một lượng lớn dữ liệu.
    *   **Ví dụ:** `data_stream` tạo ra một chuỗi các số từ 0 đến 999999 mà không cần lưu trữ tất cả các số này trong bộ nhớ cùng một lúc.
    *   **Cú pháp:**
        ```python
        def generator_name(arguments):
            for item in iterable:
                yield item
        ```
*   **Context Manager (`with`):** Đảm bảo rằng các tài nguyên được giải phóng đúng cách sau khi sử dụng.
    *   **Giải thích:** Context manager là một đối tượng định nghĩa các hành động cần thực hiện khi bắt đầu và kết thúc một khối code. Context manager thường được sử dụng với từ khóa `with` để đảm bảo rằng các tài nguyên như file, kết nối mạng, ... được giải phóng đúng cách sau khi sử dụng, ngay cả khi có lỗi xảy ra.
    *   **Ví dụ:** `with open("file.txt", "r") as f:` đảm bảo rằng file `file.txt` sẽ được đóng sau khi đọc xong, ngay cả khi có lỗi xảy ra.
    *   **Cú pháp:**
        ```python
        with context_manager as variable:
            # code block
        ```
*   **OOP nâng cao (kế thừa, `@dataclass`, `typing`, `__str__`, `__repr__`):** Các khái niệm quan trọng trong lập trình hướng đối tượng giúp bạn xây dựng các ứng dụng phức tạp một cách dễ dàng hơn.
        *   **Giải thích:**
            *   **Kế thừa:** Cho phép một class kế thừa các thuộc tính và phương thức từ một class khác.
            *   `@dataclass`: Một decorator giúp tự động tạo các phương thức `__init__`, `__repr__`, `__eq__`, ... cho một class.
            *   `typing`: Module cung cấp các type hint để chỉ định kiểu dữ liệu của các biến, tham số và giá trị trả về của hàm.
            *   `__str__`: Phương thức trả về một chuỗi đại diện cho đối tượng.
            *   `__repr__`: Phương thức trả về một chuỗi đại diện cho đối tượng, thường được sử dụng cho mục đích debug.

##### **🧪 Ví dụ thực hành:**

```python
# List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Decorator đo thời gian chạy hàm
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
def slow_func():
    time.sleep(1)
    return "Done"

# Generator
def data_stream():
    for i in range(1000000):
        yield i
```

**📚 Tham khảo:**

*   [Python List Comprehension](https://realpython.com/list-comprehension-python/)
*   [Python Lambda Functions](https://realpython.com/python-lambda/)
*   [Python Decorators](https://realpython.com/primer-on-python-decorators/)
*   [Python Generators](https://realpython.com/python-generators/)
*   [Python Context Managers](https://realpython.com/python-with-statement/)
*   [Python Dataclasses](https://realpython.com/python-data-classes/)

---

#### 3. OOP + Type Hint + Dataclass

**Giải thích:**

*   **OOP (Object-Oriented Programming):** Lập trình hướng đối tượng là một phương pháp lập trình dựa trên khái niệm "đối tượng", chứa dữ liệu (thuộc tính) và code (phương thức) để thao tác dữ liệu đó.
    *   **Ví dụ:** Trong Python, mọi thứ đều là đối tượng, từ số nguyên đến chuỗi đến hàm.
*   **Type Hint:** Type Hint là một tính năng mới trong Python 3.5+ cho phép bạn chỉ định kiểu dữ liệu của các biến, tham số và giá trị trả về của hàm. Điều này giúp code dễ đọc hơn và giúp bạn phát hiện lỗi sớm hơn.
    *   **Ví dụ:** `x: int = 10` chỉ định rằng biến `x` có kiểu dữ liệu là số nguyên.
*   **Dataclass:** Dataclass là một decorator trong Python 3.7+ giúp bạn tự động tạo các phương thức `__init__`, `__repr__`, `__eq__`, ... cho một class. Điều này giúp bạn viết code ngắn gọn hơn và dễ bảo trì hơn.
    *   **Ví dụ:** `@dataclass` giúp bạn không cần phải viết phương thức `__init__` cho class `ModelConfig`.

**Ví dụ:**

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    model_name: str
    features: List[str]
    target: str

config = ModelConfig("LinearRegression", ["x1", "x2"], "y")
```

Trong ví dụ này, chúng ta sử dụng `@dataclass` để tạo một class `ModelConfig` với các thuộc tính `model_name`, `features` và `target`. Chúng ta cũng sử dụng `typing.List` để chỉ định rằng thuộc tính `features` là một list các string.

**📚 Tham khảo:**

*   [Python OOP](https://realpython.com/python3-object-oriented-programming/)
*   [Python Type Hint](https://realpython.com/python-type-checking/)
*   [Python Dataclasses](https://realpython.com/python-data-classes/)
*   [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/)
*   [PEP 526 -- Syntax for Variable Annotations](https://peps.python.org/pep-0526/)
*   [PEP 557 -- Data Classes](https://peps.python.org/pep-0557/)

---

#### 4. Git & GitHub Workflow chuyên nghiệp

**✅ Kiến thức chính:**

*   **Git init, branch, commit, push, pull, merge:** Các lệnh cơ bản để quản lý code bằng Git.
    *   `git init`: Khởi tạo một Git repository mới. Lệnh này tạo một thư mục `.git` ẩn trong thư mục dự án của bạn. Thư mục này chứa tất cả các thông tin về lịch sử thay đổi của dự án.
    *   `git branch`: Tạo, liệt kê hoặc xóa các branch. Branch là một nhánh phát triển song song với nhánh chính (thường là `main` hoặc `master`). Bạn có thể sử dụng branch để phát triển các tính năng mới hoặc sửa lỗi mà không ảnh hưởng đến nhánh chính.
    *   `git commit`: Lưu các thay đổi vào repository. Commit là một bản ghi lại các thay đổi bạn đã thực hiện trong dự án. Mỗi commit có một message mô tả các thay đổi đó.
    *   `git push`: Đẩy các thay đổi lên remote repository (ví dụ: GitHub). Sau khi bạn đã commit các thay đổi của mình, bạn cần đẩy chúng lên remote repository để chia sẻ với người khác.
    *   `git pull`: Kéo các thay đổi từ remote repository về local. Khi người khác đã đẩy các thay đổi lên remote repository, bạn cần kéo chúng về local để cập nhật dự án của mình.
    *   `git merge`: Hợp nhất các thay đổi từ một branch vào branch hiện tại. Khi bạn đã hoàn thành việc phát triển một tính năng mới hoặc sửa lỗi trên một branch, bạn cần hợp nhất các thay đổi đó vào nhánh chính.
*   **PR (pull request), review, squash merge:** Quy trình làm việc nhóm hiệu quả trên GitHub.
    *   `PR (Pull Request)`: Một yêu cầu để hợp nhất các thay đổi từ một branch vào một branch khác. Pull request là một cách để bạn yêu cầu người khác xem xét code của bạn trước khi hợp nhất nó vào nhánh chính.
    *   `Review`: Quá trình xem xét code của người khác trước khi hợp nhất. Review giúp đảm bảo rằng code được viết đúng cách và tuân thủ các quy tắc và tiêu chuẩn của dự án.
    *   `Squash merge`: Hợp nhất nhiều commit thành một commit duy nhất. Squash merge giúp giữ cho lịch sử commit của dự án sạch sẽ và dễ đọc.
*   **Viết README chuẩn, dùng GitHub Actions CI:** Các best practices để tạo một dự án chuyên nghiệp trên GitHub.
    *   `README`: Một file mô tả dự án, cách sử dụng, ... File README là file đầu tiên mà người khác sẽ nhìn thấy khi họ truy cập dự án của bạn trên GitHub. Vì vậy, nó rất quan trọng để viết một file README rõ ràng và đầy đủ thông tin.
    *   `GitHub Actions CI`: Một công cụ để tự động build, test và deploy code. GitHub Actions CI giúp bạn tự động hóa các tác vụ như build, test và deploy code. Điều này giúp bạn tiết kiệm thời gian và đảm bảo rằng code của bạn luôn hoạt động đúng cách.

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

#### 5. Thiết lập pre-commit để tự động format code

**Giải thích:**

`pre-commit` là một framework giúp quản lý các hook trước khi commit code. Các hook này có thể được sử dụng để tự động format code, kiểm tra lỗi chính tả, ... Điều này giúp đảm bảo rằng code của bạn luôn tuân thủ các quy tắc và tiêu chuẩn của dự án.

**Hướng dẫn:**

1.  Cài đặt `pre-commit`:

```bash
pip install pre-commit black flake8 isort
pre-commit install
```

    *   `pip install pre-commit black flake8 isort`: Lệnh này cài đặt các thư viện `pre-commit`, `black`, `flake8` và `isort`.
    *   `pre-commit install`: Lệnh này cài đặt các hook `pre-commit` vào repository của bạn.

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

File `.pre-commit-config.yaml` này định nghĩa các hook sẽ được chạy trước khi commit code. Trong ví dụ này, chúng ta sử dụng 3 hook:

*   `black`: Tự động format code Python theo chuẩn PEP 8.
    *   `black`: Một công cụ format code Python theo chuẩn PEP 8.
*   `flake8`: Kiểm tra lỗi code Python.
    *   `flake8`: Một công cụ kiểm tra lỗi code Python.
*   `isort`: Sắp xếp các import trong code Python.
    *   `isort`: Một công cụ sắp xếp các import trong code Python.

**📚 Tham khảo:**

*   [pre-commit documentation](https://pre-commit.com/)
*   [black documentation](https://github.com/psf/black)
*   [flake8 documentation](https://flake8.pycqa.org/en/latest/)
*   [isort documentation](https://pycqa.github.io/isort/)

---

### 🧪 Bài Lab Day 1

**Hướng dẫn:**

Trong bài lab này, bạn sẽ thực hành các kiến thức đã học trong ngày 1.

1.  **Cài Conda + VS Code + JupyterLab + Tạo folder cấu trúc project ML:** Làm theo hướng dẫn ở phần 1 để cài đặt môi trường phát triển.
2.  **Tạo file `main.py` và viết class `MLPipeline` sử dụng `@dataclass`:** Tạo một file `main.py` trong thư mục `src` và viết một class `MLPipeline` sử dụng `@dataclass` để định nghĩa cấu trúc của một pipeline ML.
3.  **Viết thử decorator + generator demo trong `src/utils.py`:** Tạo một file `utils.py` trong thư mục `src` và viết một decorator và một generator để demo các kiến thức đã học.
4.  **Tạo `.git` repo, commit, push lên GitHub:** Tạo một Git repository cho dự án của bạn và commit các thay đổi lên GitHub.
5.  **Tích hợp `pre-commit` để format code tự động:** Làm theo hướng dẫn ở phần 5 để tích hợp `pre-commit` vào dự án của bạn.

---

### 📝 Bài tập về nhà Day 1:

1.  **Viết 3 đoạn code nhỏ:**

    *   Dùng list comprehension lọc và xử lý danh sách số nguyên: Ví dụ, lọc các số chẵn từ một list các số nguyên.
    *   Một hàm có dùng `@decorator` tự viết: Ví dụ, một decorator để log thông tin về hàm được gọi.
    *   Một generator yield ra 5 số lẻ đầu tiên lớn hơn 100: Sử dụng `yield` để tạo ra các số lẻ một cách tuần tự.

2.  **Đọc thêm về các chủ đề: `@dataclass`, `with` context manager, `typing`:** Tìm hiểu sâu hơn về các khái niệm này để hiểu rõ hơn về cách chúng hoạt động và cách sử dụng chúng trong code.

3.  **Cấu hình VS Code format code tự động khi save file:** Tìm hiểu cách cấu hình VS Code để tự động format code khi bạn lưu file. Điều này giúp bạn duy trì một style code nhất quán trong dự án của mình.

4.  **Commit thay đổi bài tập vào repo GitHub đã tạo:** Commit các thay đổi của bạn lên GitHub để lưu lại và chia sẻ với người khác.
