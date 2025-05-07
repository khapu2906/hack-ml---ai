
## 📚 Buổi 11 – GAN và Stable Diffusion

### 🎯 Mục tiêu:

*   Hiểu về **Generative Adversarial Networks (GANs)** và cách chúng hoạt động.
*   Tìm hiểu về **Stable Diffusion** và ứng dụng trong tạo hình ảnh từ văn bản.
*   Nắm vững các kỹ thuật tạo hình ảnh và video bằng cách sử dụng các mô hình học sâu.

---

### 🔍 Nội dung chính:

#### 1. **Giới thiệu về GAN (Generative Adversarial Networks):**

*   **GAN** là một mô hình học sâu gồm hai mạng nơ-ron: **Generator** và **Discriminator**.

*   **Generator**: Tạo ra dữ liệu giả (ví dụ, hình ảnh), cố gắng làm cho dữ liệu giả này trông giống dữ liệu thật.

*   **Discriminator**: Nhận dữ liệu và phân biệt giữa dữ liệu thật và dữ liệu giả do Generator tạo ra.

*   Quá trình huấn luyện GAN là một trò chơi đối kháng giữa Generator và Discriminator:

    *   Generator cố gắng lừa Discriminator.
    *   Discriminator cố gắng phân biệt dữ liệu thật và giả.

*   **Mục tiêu**: Tìm được điểm cân bằng, khi đó Generator tạo ra dữ liệu không thể phân biệt được với dữ liệu thật.

*   **Công thức loss function của GAN**:

    *   Generator: $L_G = -\mathbb{E}_x[\log D(G(x))]$
    *   Discriminator: $L_D = -\mathbb{E}_x[\log D(x)] - \mathbb{E}_z[\log(1 - D(G(z)))]$

**Giải thích:**

*   **GAN:** Mạng đối kháng sinh tạo, gồm hai mạng Generator và Discriminator.
*   **Generator:** Mạng sinh dữ liệu giả.
*   **Discriminator:** Mạng phân biệt dữ liệu thật và giả.
*   **Quá trình huấn luyện:** Trò chơi đối kháng giữa Generator và Discriminator.
*   **Mục tiêu:** Cân bằng giữa Generator và Discriminator.
*   **Công thức loss function:** Công thức toán học để tính loss của Generator và Discriminator.

#### 2. **Cấu trúc cơ bản của GAN:**

*   **Generator**:

    *   Mạng nơ-ron có nhiệm vụ sinh ra hình ảnh từ một vector ngẫu nhiên (z).
*   **Discriminator**:

    *   Mạng nơ-ron có nhiệm vụ phân biệt giữa dữ liệu thật và dữ liệu giả.

**Giải thích:**

*   **Generator:** Mạng sinh dữ liệu từ vector ngẫu nhiên.
*   **Discriminator:** Mạng phân biệt dữ liệu thật và giả.

#### 3. **Các ứng dụng của GAN:**

*   **Tạo hình ảnh**: GAN có thể tạo ra hình ảnh giả giống với hình ảnh thật (Ví dụ: tạo ra các bức tranh, ảnh chân dung).
*   **Tạo video**: GAN có thể tạo ra các video ngắn từ các hình ảnh liên tiếp.
*   **Học mô hình ngược**: GAN có thể học từ các dữ liệu đầu vào để tái tạo lại dữ liệu hoặc cải thiện chất lượng của dữ liệu đó.

**Giải thích:**

*   **Tạo hình ảnh:** Ứng dụng của GAN trong việc tạo ra hình ảnh mới.
*   **Tạo video:** Ứng dụng của GAN trong việc tạo ra video mới.
*   **Học mô hình ngược:** Ứng dụng của GAN trong việc tái tạo hoặc cải thiện dữ liệu.

#### 4. **Giới thiệu về Stable Diffusion:**

*   **Stable Diffusion** là một mô hình học sâu nổi bật trong việc tạo ra hình ảnh chất lượng cao từ mô tả văn bản (text-to-image).
*   Mô hình này dựa trên kỹ thuật **Diffusion Models**, một dạng biến thể của mạng GAN, cho phép mô hình học từ dữ liệu và tạo ra hình ảnh một cách mượt mà, không có hiện tượng bị mất chất lượng trong quá trình tạo.
*   **Điểm đặc biệt** của Stable Diffusion:

    *   **Text-to-Image**: Có thể tạo hình ảnh từ mô tả văn bản.
    *   **Latent Space**: Mô hình hoạt động trong không gian ẩn, làm giảm sự phức tạp và thời gian tính toán.
    *   **Hiệu quả**: Tạo ra hình ảnh có độ phân giải cao và chi tiết hơn so với các mô hình khác.

**Giải thích:**

*   **Stable Diffusion:** Mô hình tạo ảnh từ văn bản dựa trên Diffusion Models.
*   **Text-to-Image:** Khả năng tạo ảnh từ mô tả văn bản.
*   **Latent Space:** Không gian ẩn giúp giảm độ phức tạp tính toán.
*   **Hiệu quả:** Tạo ra ảnh chất lượng cao.

#### 5. **Ứng dụng của Stable Diffusion:**

*   **Text-to-Image generation**: Tạo ra hình ảnh từ các mô tả văn bản.
*   **Image Super-Resolution**: Tạo hình ảnh độ phân giải cao từ hình ảnh độ phân giải thấp.
*   **Inpainting**: Sửa chữa phần bị thiếu hoặc lỗi trong hình ảnh.

---

### 🧪 Bài lab Buổi 11:

#### 1. **Cài đặt một mô hình GAN cơ bản để tạo hình ảnh:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

# Cấu trúc Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

# Cấu trúc Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Khởi tạo mô hình
generator = Generator()
discriminator = Discriminator()

# Optimizer
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss function
criterion = nn.BCELoss()

# Huấn luyện mô hình GAN (Code cơ bản cho huấn luyện)
```

**Hướng dẫn:**

*   Cài đặt các thư viện cần thiết: `torch`, `torchvision`.
*   Xây dựng lớp Generator và Discriminator.
*   Khởi tạo mô hình và optimizer.
*   Huấn luyện mô hình GAN.

#### 2. **Cài đặt Stable Diffusion (Text-to-Image)**:

*   Cài đặt và sử dụng **Stable Diffusion** với mô tả văn bản để tạo hình ảnh.

```bash
pip install diffusers transformers
```

```python
from diffusers import StableDiffusionPipeline
import torch

# Tải mô hình Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original", torch_dtype=torch.float16)
pipe.to("cuda")

# Sử dụng mô hình tạo hình ảnh từ văn bản
prompt = "A beautiful sunset over the ocean"
image = pipe(prompt).images[0]
image.show()
```

**Hướng dẫn:**

*   Cài đặt các thư viện cần thiết: `diffusers`, `transformers`.
*   Tải mô hình Stable Diffusion.
*   Sử dụng mô hình để tạo hình ảnh từ văn bản.

---

### 📝 Bài tập về nhà Buổi 11:

1.  **Xây dựng GAN tạo hình ảnh**:

    *   Tạo một mô hình GAN và huấn luyện nó để sinh ra hình ảnh đơn giản (ví dụ: hình ảnh chữ số từ bộ dữ liệu MNIST).
    *   So sánh kết quả với các mô hình khác như MLP.
2.  **Tạo hình ảnh từ mô tả văn bản bằng Stable Diffusion**:

    *   Sử dụng mô hình Stable Diffusion để tạo ra hình ảnh từ các mô tả văn bản của bạn.
    *   Chỉnh sửa prompt để tạo ra các hình ảnh sáng tạo khác nhau.
3.  **Nghiên cứu các ứng dụng khác của GAN**:

    *   Tìm hiểu thêm về ứng dụng của GAN trong lĩnh vực như video generation, data augmentation, và transfer learning.

