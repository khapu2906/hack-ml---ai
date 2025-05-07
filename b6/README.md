### ✅ **Day 6 – Xây REST API dự đoán với FastAPI**

**🎯 Mục tiêu:** Deploy mô hình Machine Learning thành một API có thể gọi từ frontend hoặc các hệ thống khác.

---

#### 1. Cài đặt FastAPI & Uvicorn

```bash
pip install fastapi uvicorn[standard] joblib
```

**Giải thích:**

*   `fastapi`: Thư viện để xây dựng API.
*   `uvicorn`: ASGI server để chạy API.
*   `joblib`: Thư viện để lưu và tải mô hình.

**📚 Tham khảo:**

*   [FastAPI documentation](https://fastapi.tiangolo.com/)
*   [Uvicorn documentation](https://www.uvicorn.org/)
*   [Joblib documentation](https://joblib.readthedocs.io/en/latest/)

---

#### 2. Tạo API dự đoán đơn giản

**📁 Cấu trúc:**

```
src/
├── api/
│   └── main.py
├── models/
│   └── model.pkl
```

**📄 `main.py`:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(input_array)[0]
    return {"prediction": int(prediction)}
```

**Giải thích:**

*   `FastAPI()`: Khởi tạo ứng dụng FastAPI.
*   `InputData`: Lớp Pydantic để định nghĩa cấu trúc dữ liệu đầu vào.
*   `@app.post("/predict")`: Định nghĩa một endpoint POST tại đường dẫn `/predict`.
*   `model.predict()`: Sử dụng mô hình đã tải để dự đoán.

**📚 Tham khảo:**

*   [FastAPI documentation](https://fastapi.tiangolo.com/)
*   [Pydantic documentation](https://pydantic-docs.helpmanual.io/)

---

#### 3. Chạy thử API

```bash
uvicorn src.api.main:app --reload
```

*   Mở docs tại: [http://localhost:8000/docs](http://localhost:8000/docs)
*   Gửi thử POST request bằng Swagger UI hoặc Postman

**Giải thích:**

*   `uvicorn`: Chạy ứng dụng FastAPI.
*   `src.api.main:app`: Đường dẫn đến ứng dụng FastAPI.
*   `--reload`: Tự động tải lại ứng dụng khi có thay đổi.

---

#### 4. Bảo vệ API với CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ định domain cụ thể
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Giải thích:**

*   `CORSMiddleware`: Middleware để xử lý CORS (Cross-Origin Resource Sharing).
*   `allow_origins`: Danh sách các origin được phép truy cập API.
*   `allow_methods`: Danh sách các HTTP method được phép.
*   `allow_headers`: Danh sách các HTTP header được phép.

**📚 Tham khảo:**

*   [FastAPI documentation - CORS](https://fastapi.tiangolo.com/middleware/cors/)

---

#### 5. Tách riêng logic ra `src/services/predict.py`

```python
# src/services/predict.py
import joblib
import numpy as np

model = joblib.load("models/model.pkl")

def predict_proba(data: list):
    return model.predict_proba([data])[0].tolist()
```

**Trong `main.py` gọi lại:**

```python
from src.services.predict import predict_proba
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict_proba")
def predict_with_prob(data: InputData):
    input_list = [data.feature1, data.feature2, data.feature3, data.feature4]
    prob = predict_proba(input_list)
    return {"probabilities": prob}
```

**Giải thích:**

*   Tách logic dự đoán ra một module riêng giúp code dễ bảo trì và tái sử dụng hơn.
*   `predict_proba()`: Hàm dự đoán xác suất của các lớp.

---

### 🧪 Bài Lab Day 6

1.  Load mô hình `.pkl` trong FastAPI
2.  Tạo POST endpoint `/predict` trả về kết quả dự đoán
3.  Tạo thêm `/predict_proba` trả về xác suất
4.  Viết file `.env` để quản lý biến môi trường (nếu cần)
5.  Ghi README hướng dẫn chạy local

**Hướng dẫn:**

*   Tạo cấu trúc thư mục như đã mô tả ở trên.
*   Viết code cho `main.py` và `src/services/predict.py`.
*   Sử dụng `joblib.load()` để tải mô hình.
*   Sử dụng `@app.post()` để định nghĩa các endpoint.
*   Sử dụng `BaseModel` để định nghĩa cấu trúc dữ liệu đầu vào.

---

### 📝 Bài tập về nhà Day 6:

1.  Tạo API `/health` để check server sống
2.  Viết unit test cho hàm `predict_proba`
3.  Tạo file `Dockerfile` để chuẩn bị cho buổi deploy
4.  Tổ chức lại mã nguồn theo mô hình MVC nhỏ gọn
5.  Commit và đẩy GitHub, chụp ảnh gửi nhóm

**Hướng dẫn:**

*   Tạo endpoint `/health` trả về status code 200.
*   Sử dụng thư viện `pytest` để viết unit test cho hàm `predict_proba`.
*   Tạo file `Dockerfile` để đóng gói ứng dụng vào container.
*   Tổ chức lại mã nguồn theo mô hình MVC (Model-View-Controller) để code dễ bảo trì hơn.
