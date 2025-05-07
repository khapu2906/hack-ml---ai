## 📚 Buổi 10 – Convolutional Neural Networks (CNN)

### 🎯 Mục tiêu:

*   Hiểu về các mạng nơ-ron tích chập (CNN) và ứng dụng của chúng trong nhận dạng hình ảnh.
*   Nắm vững các lớp cơ bản trong CNN, như Convolution, Pooling, và Fully Connected.
*   Tìm hiểu cách CNN được sử dụng trong các bài toán nhận dạng ảnh và phân loại.

---

### 🔍 Nội dung chính:

#### 1. **Giới thiệu về CNN:**

*   **Convolutional Neural Networks** (CNN) là một loại mạng nơ-ron đặc biệt hiệu quả trong việc xử lý dữ liệu có dạng lưới, như hình ảnh (2D) hoặc video (3D).
*   CNN được sử dụng phổ biến trong các bài toán như nhận dạng đối tượng trong ảnh, phân loại ảnh, và nhận diện khuôn mặt.
*   CNN được đặc trưng bởi ba loại lớp chính: Convolutional, Pooling và Fully Connected.

**Giải thích:**

*   **Convolutional Neural Networks (CNN):** Mạng nơ-ron tích chập, hiệu quả trong xử lý dữ liệu có cấu trúc lưới.
*   **Ứng dụng:** Các ứng dụng phổ biến của CNN.
*   **Các lớp chính:** Các lớp cơ bản tạo nên CNN.

#### 2. **Lớp Convolution:**

*   Lớp Convolution thực hiện phép nhân ma trận giữa một kernel (hoặc filter) và vùng tương ứng trong ảnh đầu vào.
*   **Chức năng**: Trích xuất các đặc trưng (features) như các cạnh, góc, hoặc kết cấu.
*   Công thức tính toán Convolution:

    $$
    \text{Output} = \sum_{i=1}^{k} (W_i * X_i)
    $$

    Trong đó, $W_i$ là các bộ lọc và $X_i$ là các phần tử ảnh tại các vị trí tương ứng.

**Giải thích:**

*   **Lớp Convolution:** Lớp thực hiện phép tích chập giữa kernel và ảnh đầu vào.
*   **Chức năng:** Trích xuất các đặc trưng từ ảnh.
*   **Công thức tính toán:** Công thức toán học của phép tích chập.

#### 3. **Lớp Pooling:**

*   Lớp Pooling giảm độ phân giải không gian của hình ảnh, giúp giảm thiểu số lượng thông tin cần xử lý.
*   Hai loại chính: **Max Pooling** (chọn giá trị lớn nhất trong một vùng) và **Average Pooling** (tính giá trị trung bình trong một vùng).

**Giải thích:**

*   **Lớp Pooling:** Lớp giảm kích thước của ảnh.
*   **Max Pooling:** Chọn giá trị lớn nhất trong một vùng.
*   **Average Pooling:** Tính giá trị trung bình trong một vùng.

#### 4. **Lớp Fully Connected (FC):**

*   Sau khi các đặc trưng đã được trích xuất qua các lớp Convolution và Pooling, dữ liệu được chuyển qua các lớp Fully Connected để thực hiện phân loại.
*   Mỗi neuron trong lớp FC liên kết với tất cả các neuron của lớp trước đó.

**Giải thích:**

*   **Lớp Fully Connected:** Lớp kết nối đầy đủ, sử dụng các đặc trưng đã trích xuất để phân loại.

#### 5. **Kỹ thuật Data Augmentation:**

*   Để cải thiện độ chính xác của CNN, kỹ thuật **Data Augmentation** có thể được sử dụng để tạo ra nhiều mẫu dữ liệu khác nhau từ cùng một hình ảnh gốc (ví dụ: xoay, thay đổi tỷ lệ, cắt ghép).

**Giải thích:**

*   **Data Augmentation:** Kỹ thuật tăng cường dữ liệu bằng cách tạo ra các biến thể của hình ảnh gốc.

#### 6. **Ứng dụng của CNN:**

*   **Nhận diện đối tượng trong ảnh**: Phát hiện và phân loại các đối tượng trong ảnh (ví dụ: nhận dạng khuôn mặt, phân loại ảnh động vật, v.v.).
*   **Phân loại ảnh**: CNN có thể được sử dụng để phân loại các ảnh thành các nhãn khác nhau.
*   **Nhận diện văn bản**: CNN có thể nhận diện văn bản trong ảnh (OCR).

**Giải thích:**

*   **Nhận diện đối tượng trong ảnh:** Ứng dụng của CNN trong việc phát hiện và phân loại đối tượng.
*   **Phân loại ảnh:** Ứng dụng của CNN trong việc phân loại ảnh.
*   **Nhận diện văn bản:** Ứng dụng của CNN trong việc nhận diện văn bản trong ảnh.

---

### 🧪 Bài lab Buổi 10:

#### 1. **Cài đặt CNN đơn giản để phân loại ảnh CIFAR-10:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Tải dữ liệu CIFAR-10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
testloader = DataLoader(testset, batch_size=4, shuffle=False)

# Định nghĩa mô hình CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Lớp convolution đầu tiên
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # Lớp convolution thứ hai
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 lớp output cho 10 loại

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Kết hợp Convolution và ReLU
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # Biến đổi thành vector 1D
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = F.relu(self.fc2(x))  # Fully connected layer 2
        x = self.fc3(x)  # Output layer
        return x

# Khởi tạo mô hình và optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Huấn luyện mô hình
for epoch in range(2):  # Loop qua nhiều epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # In mỗi 2000 bước
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

**Hướng dẫn:**

*   Cài đặt các thư viện cần thiết: `torch`, `torchvision`.
*   Tải bộ dữ liệu CIFAR-10.
*   Định nghĩa mô hình CNN.
*   Khởi tạo mô hình và optimizer.
*   Huấn luyện mô hình.

#### 2. **Cải thiện mô hình với Data Augmentation:**

```python
transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
trainloader_augmented = DataLoader(trainset_augmented, batch_size=4, shuffle=True)

# Huấn luyện lại với bộ dữ liệu augmented
```

**Hướng dẫn:**

*   Tạo một transform để thực hiện data augmentation.
*   Tải lại bộ dữ liệu CIFAR-10 với transform mới.
*   Huấn luyện lại mô hình với bộ dữ liệu đã được augmented.

---

#### 7. Các kiến trúc CNN phổ biến:

*   **LeNet-5:**
    *   **Giải thích:** Một trong những kiến trúc CNN đầu tiên, được thiết kế để nhận diện chữ số viết tay.
    *   **Ưu điểm:** Đơn giản, dễ hiểu.
    *   **Nhược điểm:** Không hiệu quả với các bài toán phức tạp hơn.
*   **AlexNet:**
    *   **Giải thích:** Một kiến trúc CNN sâu hơn LeNet-5, đã đạt được kết quả ấn tượng trong cuộc thi ImageNet.
    *   **Ưu điểm:** Sử dụng ReLU activation function và dropout để giảm overfitting.
    *   **Nhược điểm:** Cấu trúc vẫn còn khá đơn giản so với các kiến trúc hiện đại.
*   **VGGNet:**
    *   **Giải thích:** Kiến trúc CNN sâu với các lớp convolution nhỏ (3x3).
    *   **Ưu điểm:** Cấu trúc đồng nhất, dễ mở rộng.
    *   **Nhược điểm:** Số lượng tham số lớn, đòi hỏi nhiều tài nguyên tính toán.
*   **ResNet:**
    *   **Giải thích:** Kiến trúc CNN sử dụng các kết nối tắt (skip connections) để giải quyết vấn đề vanishing gradient trong các mạng sâu.
    *   **Ưu điểm:** Cho phép huấn luyện các mạng rất sâu, đạt được kết quả tốt trên nhiều bài toán.
    *   **Nhược điểm:** Cấu trúc phức tạp hơn so với các kiến trúc trước đó.
*   **InceptionNet:**
    *   **Giải thích:** Kiến trúc CNN sử dụng các module Inception để trích xuất các đặc trưng ở nhiều kích thước khác nhau.
    *   **Ưu điểm:** Hiệu quả trong việc trích xuất các đặc trưng đa dạng.
    *   **Nhược điểm:** Cấu trúc phức tạp.

---

### 📝 Bài tập về nhà Buổi 10:

1.  **Cài đặt CNN cho bộ dữ liệu MNIST**:

    *   Tạo một mô hình CNN cho bộ dữ liệu MNIST và thực hiện phân loại các chữ số viết tay. So sánh kết quả với một mô hình MLP (Multi-Layer Perceptron).
2.  **Cải thiện mô hình CNN**:

    *   Áp dụng kỹ thuật Data Augmentation cho bộ dữ liệu CIFAR-10 và quan sát sự thay đổi về độ chính xác so với mô hình không có augmentation.
3.  **Tìm hiểu các kiến trúc CNN khác**:

    *   Tìm hiểu về các mô hình CNN phổ biến như VGG, ResNet và Inception. So sánh các kiến trúc này với mô hình CNN đơn giản.
