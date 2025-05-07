### ✅ **Day 7 – Deploy ML Model lên Cloud miễn phí**

**🎯 Mục tiêu:** Biến mô hình Machine Learning của bạn thành một sản phẩm có thể truy cập từ bất kỳ đâu trên Internet.

---

#### 1. Chuẩn bị Dockerfile

**📄 `Dockerfile`:**

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Giải thích:**

*   `FROM python:3.10-slim`: Sử dụng image Python 3.10 slim.
*   `WORKDIR /app`: Đặt thư mục làm việc là `/app`.
*   `COPY requirements.txt .`: Copy file `requirements.txt` vào thư mục làm việc.
*   `RUN pip install --no-cache-dir -r requirements.txt`: Cài đặt các thư viện từ `requirements.txt`.
*   `COPY . .`: Copy toàn bộ source code vào thư mục làm việc.
*   `CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]`: Chạy ứng dụng FastAPI bằng Uvicorn.

**📄 `requirements.txt`:**

```
fastapi
uvicorn[standard]
scikit-learn
joblib
numpy
```

**Giải thích:**

*   Liệt kê các thư viện cần thiết cho ứng dụng.

---

#### 2. Triển khai với [Render.com](https://render.com)

*   Tạo tài khoản (có thể dùng GitHub login)
*   Tạo dịch vụ mới → Web Service
*   Kết nối repo GitHub có project
*   Thiết lập:

    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
    *   **Python Version:** 3.10
    *   **Port:** 8000
*   Chờ vài phút, truy cập domain được cung cấp

**Giải thích:**

*   Hướng dẫn triển khai ứng dụng lên Render.com.

---

#### 3. Triển khai với Hugging Face Spaces (Gradio + FastAPI)

*   Cài thêm `gradio`, tạo giao diện đơn giản:

```bash
pip install gradio
```

**📄 `gradio_app.py`:**

```python
import gradio as gr
import joblib
import numpy as np

model = joblib.load("models/model.pkl")

def predict(f1, f2, f3, f4):
    x = np.array([[f1, f2, f3, f4]])
    return int(model.predict(x)[0])

iface = gr.Interface(
    fn=predict,
    inputs=["number", "number", "number", "number"],
    outputs="number"
)

iface.launch()
```

**Giải thích:**

*   `gradio`: Thư viện để tạo giao diện người dùng đơn giản.
*   `gr.Interface()`: Tạo giao diện người dùng cho hàm `predict()`.
*   `iface.launch()`: Chạy ứng dụng Gradio.

*   Push lên GitHub
*   Truy cập [hf.co/spaces](https://huggingface.co/spaces) → Create New Space → chọn Gradio + repo
*   Deploy và chia sẻ link

**📚 Tham khảo:**

*   [Gradio documentation](https://www.gradio.app/docs/interface)

---

#### 4. Tùy chọn: Deta Space (Free, nhẹ, nhanh)

*   Truy cập: [https://deta.space](https://deta.space/)
*   Tạo "Micro" app → upload zip hoặc link GitHub
*   Yêu cầu `main.py` nằm ở root
*   Tự động tạo API endpoint

**Giải thích:**

*   Hướng dẫn triển khai ứng dụng lên Deta Space.

---

### 🧪 Bài Lab Day 7

1.  Tạo Dockerfile, build image và test local
2.  Push code lên GitHub
3.  Deploy lên Render hoặc Hugging Face Spaces
4.  Test API online bằng Postman hoặc frontend đơn giản
5.  Chia sẻ link endpoint với bạn bè

**Hướng dẫn:**

*   Thực hiện theo các bước đã mô tả ở trên để tạo Dockerfile, build image, push code lên GitHub, và deploy ứng dụng lên Render hoặc Hugging Face Spaces.
*   Sử dụng Postman hoặc một frontend đơn giản để test API online.
*   Chia sẻ link endpoint với bạn bè để họ có thể trải nghiệm ứng dụng của bạn.

---

### 📝 Bài tập về nhà Day 7:

1.  Viết README.md hướng dẫn chạy project + cách deploy
2.  Ghi lại video demo sử dụng API hoặc Gradio app
3.  Nghiên cứu thêm: `ngrok`, `Railway.app`, `Fly.io` để deploy miễn phí
4.  Đọc thêm: `CI/CD với GitHub Actions`, `GitHub Codespaces`, `API Key bảo mật`
5.  Gửi link project + video demo lên nhóm

**Hướng dẫn:**

*   Viết một file README.md chi tiết hướng dẫn cách chạy project và deploy ứng dụng lên các nền tảng khác nhau.
*   Ghi lại một video demo sử dụng API hoặc Gradio app để giới thiệu ứng dụng của bạn.
*   Nghiên cứu thêm về các nền tảng deploy miễn phí khác như `ngrok`, `Railway.app`, `Fly.io`.
*   Đọc thêm về các chủ đề liên quan đến CI/CD, GitHub Codespaces, và bảo mật API.
