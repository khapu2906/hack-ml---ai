## 📚 Buổi 12 – Các mô hình Generative khác: VAE, Diffusion Models và ứng dụng trong nghệ thuật tạo hình

### 🎯 Mục tiêu:

*   Hiểu và nắm bắt các mô hình Generative khác ngoài GAN, bao gồm **Variational Autoencoders (VAE)** và **Diffusion Models**.
*   Nắm bắt các ứng dụng của các mô hình này trong nghệ thuật tạo hình và các lĩnh vực khác.
*   Làm quen với cách sử dụng các mô hình học sâu trong việc tạo ra hình ảnh và video từ dữ liệu có sẵn.

---

### 🔍 Nội dung chính:

#### 1. **Giới thiệu về Variational Autoencoders (VAE):**

*   **Autoencoders**: Mô hình học không giám sát học cách nén dữ liệu vào không gian ẩn và tái tạo lại dữ liệu từ không gian này.
*   **Variational Autoencoders (VAE)** là phiên bản cải tiến của Autoencoder, sử dụng lý thuyết thống kê để điều khiển không gian ẩn sao cho mô hình có thể sinh dữ liệu mới (generative model).
    *   VAE thay vì chỉ đơn giản nén dữ liệu vào một không gian ẩn cố định, nó học phân phối xác suất của không gian ẩn và mẫu từ phân phối đó để tạo ra dữ liệu mới.
*   **Cấu trúc VAE**:
    *   **Encoder**: Mạng nơ-ron chuyển đầu vào thành phân phối xác suất trong không gian ẩn (mean, variance).
    *   **Decoder**: Mạng nơ-ron chuyển từ không gian ẩn trở lại thành dữ liệu.
*   **Loss function của VAE**:
    *   **Reconstruction loss**: Đo lường độ chính xác của việc tái tạo lại dữ liệu gốc.
    *   **KL divergence**: Đo lường sự khác biệt giữa phân phối ẩn học được và phân phối chuẩn (thường là Gaussian).

**Giải thích:**

*   **Autoencoders:** Mô hình học không giám sát để nén và tái tạo dữ liệu.
*   **Variational Autoencoders (VAE):** Mô hình Generative dựa trên Autoencoders.
*   **Cấu trúc VAE:** Encoder và Decoder.
*   **Loss function của VAE:** Reconstruction loss và KL divergence.

#### 2. **Ứng dụng của VAE:**

*   **Image Generation**: Tạo ra hình ảnh từ không gian ẩn.
*   **Data Augmentation**: Tạo ra các dữ liệu mới từ phân phối xác suất.
*   **Latent Space Manipulation**: Tạo ra các thay đổi trong không gian ẩn và sinh dữ liệu mới theo các biến đổi này.

**Giải thích:**

*   **Image Generation:** Tạo hình ảnh mới.
*   **Data Augmentation:** Tăng cường dữ liệu huấn luyện.
*   **Latent Space Manipulation:** Thay đổi không gian ẩn để tạo ra các biến thể dữ liệu.

#### 3. **Giới thiệu về Diffusion Models:**

*   **Diffusion Models** là một lớp mô hình generative rất mạnh mẽ, hoạt động bằng cách "phá vỡ" dữ liệu thành nhiễu và sau đó "hồi phục" nó từ nhiễu đó.
*   Quá trình diffusion mô phỏng quá trình nhiễu hóa dần dần, sau đó là quá trình hồi phục (denoising).
*   **Các ứng dụng của Diffusion Models**:
    *   **Text-to-Image generation**: Tạo ra hình ảnh từ mô tả văn bản (Text-to-image).
    *   **Image Super-Resolution**: Tạo hình ảnh độ phân giải cao từ hình ảnh độ phân giải thấp.
    *   **Inpainting**: Sửa chữa phần bị thiếu hoặc lỗi trong hình ảnh.
*   **Ưu điểm của Diffusion Models**:
    *   Tạo ra hình ảnh chất lượng cao hơn GAN trong nhiều trường hợp.
    *   Không gặp vấn đề mode collapse như trong GANs.
    *   Quá trình học mô phỏng với sự nhiễu hóa và phục hồi dữ liệu rất hiệu quả trong việc tạo ra dữ liệu chân thực.

**Giải thích:**

*   **Diffusion Models:** Mô hình Generative dựa trên quá trình khuếch tán.
*   **Ứng dụng:** Các ứng dụng phổ biến của Diffusion Models.
*   **Ưu điểm:** Các ưu điểm so với GAN.

#### 4. **Sử dụng VAE và Diffusion Models trong Nghệ thuật Tạo Hình:**

*   **Vẽ tranh, tạo hình ảnh nghệ thuật**: Cả VAE và Diffusion Models có thể được sử dụng để tạo ra các hình ảnh nghệ thuật, từ tranh vẽ cho đến hình ảnh hiện thực.
*   **Các ứng dụng khác**: Ngoài việc tạo hình ảnh, các mô hình này còn có thể được áp dụng trong việc tạo video, chuyển đổi phong cách hình ảnh, tạo ảnh 3D từ ảnh 2D.

**Giải thích:**

*   **Vẽ tranh, tạo hình ảnh nghệ thuật:** Ứng dụng trong việc tạo ra các tác phẩm nghệ thuật.
*   **Các ứng dụng khác:** Các ứng dụng khác của VAE và Diffusion Models.

---

### 🧪 Bài lab Buổi 12:

#### 1. **Xây dựng và huấn luyện VAE để tạo hình ảnh:**

```python
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

# Cấu trúc VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # Mean
        self.fc22 = nn.Linear(400, 20)  # Log variance
        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x.view(-1, 784)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function VAE
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL divergence between Gaussian prior and posterior
    # The prior is a standard normal, so mean=0 and variance=1
    # The posterior is defined by mu and logvar (output of the encoder)
    # This gives the KL divergence between two Gaussian distributions
    # which is calculated as 0.5 * (log(sigma^2) - log(sigma_0^2) - 1 + (mu - mu_0)^2 / sigma_0^2)
    # Here, sigma_0^2 = 1 and mu_0 = 0
    # This simplifies to the equation:
    # 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    # where mu and logvar are from the encoder
    # and recon_x is the reconstructed image.
    MSE = 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - logvar - 1)
    return BCE + MSE

# Huấn luyện VAE
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dataloader cho bộ dữ liệu MNIST
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()), batch_size=128, shuffle=True)

# Huấn luyện VAE
model.train()
for epoch in range(10):
    train_loss = 0
    for data, _ in train_loader:
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch {}, Training loss {}'.format(epoch, train_loss / len(train_loader.dataset)))
```

**Hướng dẫn:**

*   Xây dựng lớp VAE với Encoder và Decoder.
*   Định nghĩa loss function cho VAE.
*   Huấn luyện VAE trên bộ dữ liệu MNIST.

#### 2. **Sử dụng Diffusion Models để tạo hình ảnh từ văn bản**:

```python
from diffusers import StableDiffusionPipeline
import torch

# Tải mô hình Diffusion
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original", torch_dtype=torch.float16)
pipe.to("cuda")

# Sử dụng mô hình tạo hình ảnh từ văn bản
prompt = "A beautiful landscape with mountains and a river"
image = pipe(prompt).images[0]
image.show()
```

**Hướng dẫn:**

*   Cài đặt thư viện `diffusers`.
*   Tải mô hình Stable Diffusion.
*   Sử dụng mô hình để tạo hình ảnh từ văn bản.

---

### 📝 Bài tập về nhà Buổi 12:

1.  **Tạo mô hình VAE của bạn**:

    *   Tạo một mô hình VAE để tạo ra hình ảnh từ bộ dữ liệu MNIST.
    *   So sánh chất lượng hình ảnh tái tạo giữa VAE và Autoencoder đơn giản.
2.  **Sử dụng Diffusion Models để tạo hình ảnh**:

    *   Sử dụng mô hình Diffusion để tạo hình ảnh từ các mô tả văn bản của bạn.
    *   Tạo ra một số hình ảnh sáng tạo từ mô tả văn bản khác nhau.
3.  **Khám phá ứng dụng nghệ thuật của Generative Models**:

    *   Tìm hiểu thêm về cách các mô hình Generative như VAE và Diffusion được sử dụng trong nghệ thuật số và tạo hình ảnh, video từ dữ liệu.
