import gzip
import numpy as np
import matplotlib.pyplot as plt
import TwoLayerNet
import functions as f
import Layers as l
import optimizers as o

# MNIST 데이터셋 로드 / 블랙박스

def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        # offset=16 은 헤더를 건너뛰고 실제 픽셀 데이터부터 읽겠다는 의미입니다.
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 28x28 크기의 이미지로 정규화하여 반환합니다.
    return data.reshape(-1, 784) / 255.0

def load_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        # offset=8 은 헤더를 건너뛰고 실제 레이블 데이터부터 읽겠다는 의미입니다.
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 파일 경로 설정 (이미지 캡처 기준)
train_img_path = 'dataset/train-images-idx3-ubyte.gz'
train_lbl_path = 'dataset/train-labels-idx1-ubyte.gz'
test_img_path = 'dataset/t10k-images-idx3-ubyte.gz'
test_lbl_path = 'dataset/t10k-labels-idx1-ubyte.gz'

# x, t 지정
x_train = load_mnist_images(train_img_path)
t_train = load_mnist_labels(train_lbl_path)
x_test = load_mnist_images(test_img_path)
t_test = load_mnist_labels(test_lbl_path)

# 데이터 확인
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)

def to_one_hot(t, num_classes=10):
    # t의 개수만큼 행, 클래스 개수만큼 열을 가진 0 행렬을 만들고
    # 정답 인덱스 위치만 1로 채웁니다.
    return np.eye(num_classes)[t]

dl = TwoLayerNet.TwoLayerNet()
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
loss_history = []

t_train = to_one_hot(t_train)
t_test = to_one_hot(t_test)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    current_loss = dl.modify(x_batch, t_batch)
    loss_history.append(current_loss)
    print(f"반복 {i+1}/{iters_num}, Loss = {current_loss} 완료")

print(dl.accuracy(x_test, t_test)*100, "%")

plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig('loss_graph.png') 
print("그래프가 loss_graph.png로 저장되었습니다.")

