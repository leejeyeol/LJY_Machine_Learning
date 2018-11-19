# -*- coding: utf-8 -*-
'''
import numpy as np

# N 는 배치 사이즈
# D_in 는 입력 차원
# H is 는 은닉 차원
# D_out 은 출력 차원
N, D_in, H, D_out = 1, 2, 8, 1

def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 인공적으로 랜덤 입력과 출력 만들기 x : 입력 y : 출력
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 가중치 초기화 하기(여러 방법이 있지만 여기서는 랜덤 초기화 사용)
# w1, w2는 가중치 행렬
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 0.1


for t in range(1000):
    # Forward pass: y를 예측
    L1 = tanh(x.dot(w1))
    L2 = sigmoid(L1.dot(w2))

    # loss 계산
    loss = np.square(L2 - y).sum() / N
    print("%d iteration Loss : %.4f" % (t, loss))

    # loss에 대한 w1,w2의 gradient를 계산하기 위해 역전파 실행
    dL2 = loss * sigmoid_derivative(L2)
    dL1 = dL2.dot(w2.T) * tanh_derivative(L1)

    grad_w2 = L1.T.dot(dL2)
    grad_w1 = L2.T.dot(dL1)

    # 역전파된 gradient를 이용해 가중치들을 업데이트
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

h = x.dot(w1)
h_tanh = tanh(h)
h2 = h_tanh.dot(w2)
y_pred = sigmoid(h2)

print(y_pred)
'''

#   XOR.py-A very simple neural network to do exclusive or.
import numpy as np

epochs = 60000  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])


def sigmoid(x): return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x): return x * (1 - x)  # derivative of sigmoid


# weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))

for i in range(epochs):
    H = sigmoid(np.dot(X, Wh))  # hidden layer results
    Z = sigmoid(np.dot(H, Wz))  # output layer results
    E = Y - Z  # how much we missed (error)
    dZ = E * sigmoid_(Z)  # delta Z
    dH = dZ.dot(Wz.T) * sigmoid_(H)  # delta H
    Wz += H.T.dot(dZ)  # update output layer weights
    Wh += X.T.dot(dH)  # update hidden layer weights

print(Z)