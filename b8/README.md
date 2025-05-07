## 📚 Buổi 8 – Nhập môn RNN và NLP

### 🎯 Mục tiêu:

*   Hiểu khái niệm về dữ liệu tuần tự và ứng dụng trong xử lý ngôn ngữ tự nhiên (NLP)
*   Nắm vững kiến trúc RNN, LSTM, GRU
*   Biết cách xử lý văn bản và xây dựng mô hình dự đoán cơ bản bằng RNN

---

### 🔍 Nội dung chính:

#### 1. **Dữ liệu tuần tự là gì?**

*   **Dữ liệu tuần tự** là loại dữ liệu có thứ tự các yếu tố (chuỗi), thông tin của một yếu tố phụ thuộc vào yếu tố trước đó.
    *   Ví dụ: Văn bản, âm thanh, chuỗi thời gian (time series).
*   **Ứng dụng**: Phân tích cảm xúc, chatbot, dịch máy, nhận diện giọng nói, nhận dạng chữ viết tay...
*   **Phân biệt dữ liệu tuần tự và dữ liệu dạng bảng**:
    *   Dữ liệu tuần tự: Thứ tự các yếu tố có ý nghĩa (ví dụ: văn bản, âm thanh).
    *   Dữ liệu dạng bảng: Mỗi dòng không có sự liên kết chặt chẽ với các dòng khác (ví dụ: dữ liệu tabular trong bảng tính).

**Giải thích:**

*   **Dữ liệu tuần tự:** Dữ liệu mà thứ tự của các phần tử có ý nghĩa quan trọng.
*   **Ứng dụng:** Các ứng dụng phổ biến của dữ liệu tuần tự.
*   **Phân biệt:** Sự khác biệt giữa dữ liệu tuần tự và dữ liệu dạng bảng.

#### 2. **Kiến trúc RNN (Recurrent Neural Network)**:

*   **RNN** là loại mạng nơ-ron có khả năng lưu trữ thông tin qua các bước thời gian (timesteps). Mỗi bước nhận đầu vào và thông tin trạng thái (hidden state) từ bước trước.
*   **Công thức cơ bản**: $h_t = f(x_t, h_{t-1})$
    *   $x_t$ là đầu vào tại thời điểm $t$
    *   $h_t$ là trạng thái ẩn (hidden state) tại thời điểm $t$
*   **Vanishing Gradient Problem**: RNN gặp khó khăn khi học dữ liệu chuỗi dài vì gradient có thể mất dần trong quá trình lan truyền ngược (backpropagation).

**Giải thích:**

*   **RNN:** Mạng nơ-ron có khả năng xử lý dữ liệu tuần tự.
*   **Công thức cơ bản:** Công thức tính toán trạng thái ẩn của RNN.
*   **Vanishing Gradient Problem:** Vấn đề khiến RNN khó học dữ liệu chuỗi dài.

#### 3. **Giải pháp: LSTM (Long Short-Term Memory) và GRU (Gated Recurrent Unit)**:

*   **LSTM**: Kiến trúc đặc biệt với các cổng (gates) để điều khiển dòng dữ liệu qua các bước thời gian. LSTM có khả năng ghi nhớ thông tin dài hạn.
*   **GRU**: Là biến thể nhẹ hơn của LSTM, sử dụng ít cổng hơn nhưng vẫn hiệu quả trong việc lưu trữ thông tin.
*   **So sánh LSTM và GRU**: LSTM có 3 cổng (input, output, forget), trong khi GRU có 2 cổng (reset, update). GRU nhanh và dễ huấn luyện hơn, nhưng LSTM có khả năng biểu diễn mối quan hệ phức tạp hơn.

**Giải thích:**

*   **LSTM:** Kiến trúc RNN cải tiến để giải quyết vấn đề vanishing gradient.
*   **GRU:** Một biến thể đơn giản hơn của LSTM.
*   **So sánh:** So sánh ưu và nhược điểm của LSTM và GRU.

#### 4. **Tokenization & Embedding trong NLP**:

*   **Tokenization**: Quá trình chuyển văn bản thành các đơn vị nhỏ hơn (tokens) như từ hoặc câu. Tokenizer giúp phân tích văn bản thành các token có thể xử lý bởi mô hình.
*   **Embedding**: Chuyển đổi các token (word) thành các vector liên tục để mô hình có thể học được các mối quan hệ giữa các từ.
    *   **One-hot encoding**: Mỗi từ là một vector với một giá trị "1" tại vị trí của từ trong từ điển, còn lại là "0". Cách này đơn giản nhưng rất thưa (sparse).
    *   **Word2Vec**: Dùng mạng nơ-ron để học các vector từ trong không gian từ vựng dựa trên ngữ cảnh xung quanh.
    *   **GloVe**: Tạo embedding từ phân tích thống kê các mối quan hệ giữa các từ trong một corpus lớn.

**Giải thích:**

*   **Tokenization:** Quá trình chia văn bản thành các đơn vị nhỏ hơn.
*   **Embedding:** Quá trình chuyển đổi các token thành vector.
*   **One-hot encoding:** Một phương pháp embedding đơn giản.
*   **Word2Vec:** Một phương pháp embedding sử dụng mạng nơ-ron.
*   **GloVe:** Một phương pháp embedding dựa trên thống kê.

#### 5. **Thử nghiệm mô hình đơn giản: Next Word Prediction**

*   Mô hình dự đoán từ tiếp theo trong câu.
*   Input: "Tôi thích" → Output: "học"
*   Cách thực hiện: Sử dụng RNN, token hóa văn bản và huấn luyện mô hình dự đoán từ tiếp theo.

**Giải thích:**

*   **Next Word Prediction:** Một bài toán NLP cơ bản.
*   **Cách thực hiện:** Mô tả các bước để xây dựng mô hình dự đoán từ tiếp theo.

---

### 🧪 Bài lab Buổi 8:

#### 1. **Tiền xử lý văn bản:**

*   Sử dụng `nltk` để loại bỏ dấu câu, chuyển về chữ thường (lowercase), và loại bỏ stopwords.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(re.sub(r'\W+', ' ', text))
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens]

print(clean_text("I am learning Natural Language Processing in Python!"))
```

**Hướng dẫn:**

*   Cài đặt thư viện `nltk`.
*   Tải các tài nguyên cần thiết của `nltk`.
*   Viết hàm `clean_text()` để thực hiện các bước tiền xử lý văn bản.

#### 2. **Tạo embedding với PyTorch:**

*   Tạo embedding cho một câu sử dụng PyTorch.

```python
import torch
import torch.nn as nn

vocab_size = 1000
embedding_dim = 50
embedding = nn.Embedding(vocab_size, embedding_dim)

sample_input = torch.tensor([1, 45, 234])
embedded = embedding(sample_input)
print(embedded.shape)  # (3, 50)
```

**Hướng dẫn:**

*   Sử dụng lớp `nn.Embedding` của PyTorch để tạo embedding.
*   Tạo một tensor đầu vào và chuyển nó qua lớp embedding.

#### 3. **Xây dựng mô hình RNN:**

*   Mô hình RNN để dự đoán từ tiếp theo.

```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)

# Dummy input: batch_size=2, sequence_length=4
x = torch.randint(0, vocab_size, (2, 4))
model = RNNModel(vocab_size=1000, embed_size=64, hidden_size=128)
output = model(x)
print(output.shape)  # (2, 4, 1000)
```

**Hướng dẫn:**

*   Tạo một lớp `RNNModel` kế thừa từ `nn.Module`.
*   Sử dụng lớp `nn.RNN` của PyTorch để tạo mô hình RNN.
*   Sử dụng lớp `nn.Linear` để tạo lớp fully connected.

---

### 📝 Bài tập về nhà Buổi 8:

1.  **Viết hàm chuẩn hóa văn bản**:

    *   Viết một hàm chuẩn hóa văn bản tiếng Việt: loại bỏ dấu câu, chuyển về lowercase, loại bỏ stopwords (có thể tự định nghĩa stopword list tiếng Việt).
2.  **So sánh kết quả embedding giữa One-Hot và Word2Vec**:

    *   Dùng `gensim` để load word2vec tiếng Việt hoặc tiếng Anh (GoogleNews).
    *   So sánh các vector embedding giữa One-Hot và Word2Vec.
3.  **Viết mô hình GRU để sinh văn bản**:

    *   Cho một đoạn văn mẫu, viết mô hình GRU đơn giản để sinh ra các từ tiếp theo.
    *   Mô hình GRU có thể sử dụng đầu vào là một chuỗi và dự đoán từ tiếp theo trong chuỗi.

