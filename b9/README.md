## 📚 Buổi 9 – Transformer & Attention

### 🎯 Mục tiêu:

*   Hiểu về kiến trúc Transformer, cơ chế Attention và ứng dụng của nó trong các mô hình NLP hiện đại như BERT, GPT.
*   Nắm vững cách thức hoạt động của Attention, Self-Attention, Multi-Head Attention.
*   Cấu trúc và cách huấn luyện mô hình Transformer.

---

### 🔍 Nội dung chính:

#### 1. **Giới thiệu về Transformer:**

*   **Transformer** là một kiến trúc mạng nơ-ron mạnh mẽ, đặc biệt trong xử lý ngôn ngữ tự nhiên (NLP).
*   Được giới thiệu trong bài báo "Attention Is All You Need" (Vaswani et al., 2017), Transformer thay thế các mạng RNN/LSTM trong nhiều ứng dụng NLP và đã giúp đạt được hiệu suất vượt trội trong các tác vụ như dịch máy, phân loại văn bản, và tạo văn bản.

**Giải thích:**

*   **Transformer:** Kiến trúc mạng nơ-ron dựa trên cơ chế Attention.
*   **"Attention Is All You Need"**: Tên bài báo giới thiệu kiến trúc Transformer.

**📚 Tham khảo:**

*   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

#### 2. **Cơ chế Attention:**

*   **Attention** cho phép mô hình tập trung vào những phần quan trọng trong dữ liệu đầu vào, thay vì xử lý tuần tự như RNN.
*   **Self-Attention** là cách mà mỗi từ trong câu có thể "chú ý" đến các từ khác trong cùng một câu. Nó giúp mô hình học được mối quan hệ giữa các từ trong câu mà không cần phải xử lý tuần tự.
*   **Scaled Dot-Product Attention**:
    *   Công thức tính Attention:

        $$
        \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
        $$
    *   $Q$: Query (truy vấn), $K$: Key (chìa khóa), $V$: Value (giá trị), $d_k$: kích thước của vector khóa.
*   **Multi-Head Attention**: Một cải tiến của Attention, cho phép mô hình tập trung vào nhiều "khía cạnh" của thông tin đầu vào cùng một lúc.

**Giải thích:**

*   **Attention:** Cơ chế cho phép mô hình tập trung vào các phần quan trọng của dữ liệu.
*   **Self-Attention:** Cơ chế Attention trong đó các từ trong câu "chú ý" lẫn nhau.
*   **Scaled Dot-Product Attention:** Công thức tính Attention.
*   **Multi-Head Attention:** Cơ chế Attention với nhiều "đầu" để tập trung vào nhiều khía cạnh khác nhau.

#### 3. **Encoder-Decoder trong Transformer:**

*   **Encoder**: Chuyển đổi câu đầu vào thành một chuỗi các đại diện (representations).
*   **Decoder**: Sử dụng chuỗi đại diện này để tạo ra câu đầu ra.
*   Mỗi Encoder và Decoder trong Transformer gồm nhiều lớp Attention và Feed Forward.

**Giải thích:**

*   **Encoder:** Phần của Transformer để mã hóa câu đầu vào.
*   **Decoder:** Phần của Transformer để giải mã và tạo ra câu đầu ra.

#### 4. **BERT & GPT – Ứng dụng của Transformer:**

*   **BERT (Bidirectional Encoder Representations from Transformers)**: Mô hình chỉ sử dụng phần Encoder của Transformer và được huấn luyện để hiểu ngữ cảnh từ cả hai phía của một từ.
*   **GPT (Generative Pre-trained Transformer)**: Mô hình chỉ sử dụng phần Decoder của Transformer, được huấn luyện để sinh văn bản từ một đoạn văn bản đầu vào.

**Giải thích:**

*   **BERT:** Mô hình Transformer sử dụng Encoder để hiểu ngữ cảnh.
*   **GPT:** Mô hình Transformer sử dụng Decoder để sinh văn bản.

**📚 Tham khảo:**

*   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
*   [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

#### 5. **Ứng dụng của Transformer trong NLP:**

*   **Dịch máy**: Chuyển ngữ giữa các ngôn ngữ (việc sử dụng cơ chế Attention giúp Transformer học được các mối quan hệ từ xa trong câu).
*   **Tạo văn bản**: Các mô hình như GPT sử dụng Transformer để tạo văn bản mới dựa trên một đoạn văn bản đầu vào.
*   **Phân loại văn bản**: Dùng Transformer để phân loại các văn bản dựa trên ngữ cảnh của các từ trong câu.

**Giải thích:**

*   **Dịch máy:** Ứng dụng của Transformer trong việc dịch ngôn ngữ.
*   **Tạo văn bản:** Ứng dụng của Transformer trong việc tạo văn bản mới.
*   **Phân loại văn bản:** Ứng dụng của Transformer trong việc phân loại văn bản.

---

### 🧪 Bài lab Buổi 9:

#### 1. **Cài đặt Attention đơn giản với PyTorch:**

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)  # Kích thước của key
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)

# Ví dụ đầu vào
Q = torch.rand(1, 3, 4)  # (batch_size, seq_len, d_k)
K = torch.rand(1, 3, 4)
V = torch.rand(1, 3, 4)

output = scaled_dot_product_attention(Q, K, V)
print(output.shape)  # (1, 3, 4)
```

**Hướng dẫn:**

*   Cài đặt hàm `scaled_dot_product_attention()` để tính Attention.
*   Tạo các tensor đầu vào `Q`, `K`, `V`.
*   Gọi hàm `scaled_dot_product_attention()` và in ra kích thước của output.

#### 2. **Multi-Head Attention trong PyTorch:**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        self.values = nn.Linear(embed_size, self.head_dim)
        self.keys = nn.Linear(embed_size, self.head_dim)
        self.queries = nn.Linear(embed_size, self.head_dim)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = values.permute(2, 0, 1, 3)
        keys = keys.permute(2, 0, 1, 3)
        query = query.permute(2, 0, 1, 3)

        energy = torch.einsum("qnhd,knhd->qnk", [query, keys])  # (num_heads, N, query_len, key_len)
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.einsum("qnk,knhd->qnhd", [attention, values]).permute(1, 2, 0, 3)
        out = out.reshape(N, query_len, self.num_heads * self.head_dim)

        out = self.fc_out(out)
        return out

# Ví dụ với dữ liệu
embedding_dim = 256
num_heads = 8
multihead_attention = MultiHeadAttention(embed_size=embedding_dim, num_heads=num_heads)

# Giả sử mỗi từ trong câu có vector embedding kích thước 256
sample_input = torch.rand(1, 5, embedding_dim)  # (batch_size, seq_len, embed_size)
output = multihead_attention(sample_input, sample_input, sample_input)
print(output.shape)  # (1, 5, 256)
```

**Hướng dẫn:**

*   Tạo một lớp `MultiHeadAttention` kế thừa từ `nn.Module`.
*   Sử dụng các lớp `nn.Linear` để tạo các ma trận trọng số cho `values`, `keys`, và `queries`.
*   Thực hiện phép tính Multi-Head Attention và in ra kích thước của output.

---

### 📝 Bài tập về nhà Buổi 9:

1.  **Cài đặt Self-Attention**:

    *   Cài đặt cơ chế Self-Attention theo công thức Scaled Dot-Product Attention cho một câu văn bản mẫu.
    *   Hãy thử nghiệm với các kích thước khác nhau của vector đầu vào.
2.  **Tạo mô hình Transformer đơn giản**:

    *   Xây dựng một mô hình Transformer sử dụng Encoder và Decoder để dự đoán một chuỗi tiếp theo từ một chuỗi đầu vào. Bạn có thể sử dụng PyTorch hoặc TensorFlow.
3.  **Tìm hiểu về BERT/GPT**:

    *   Tìm hiểu cách BERT và GPT được huấn luyện. Làm thế nào chúng sử dụng Transformer để hiểu và tạo văn bản?

