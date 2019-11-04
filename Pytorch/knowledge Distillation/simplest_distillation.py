import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

def softmax(np_data, temperature):
    result = []
    denominator = np.sum(np.exp(np_data/temperature))
    for i in range(len(np_data)):
        q = np.exp(np_data[i]/temperature)/denominator
        result.append(q)
    np.asarray(result)
    return result

artificial_logit = np.random.rand(4)
artificial_logit = np.array([0.1,0.2,2.4,0])
x = np.arange(len(artificial_logit))

print(artificial_logit)
print(np.sum(artificial_logit))
plt.bar(x,artificial_logit)
plt.title("original data(logit)")
plt.show()

# softmax
knowledge = softmax(artificial_logit,1)
print(knowledge)
print(np.sum(knowledge))
plt.bar(x,knowledge)
plt.title("original data(softmax)")
plt.show()

# softmax - temparature 2
knowledge = softmax(artificial_logit,2)
print(knowledge)
print(np.sum(knowledge))
plt.bar(x,knowledge)
plt.title("original data(softmax - temparature 2)")
plt.show()


# softmax - temparature 5
knowledge = softmax(artificial_logit,20)
print(knowledge)
print(np.sum(knowledge))
plt.bar(x,knowledge)
plt.title("original data(softmax - temparature 20)")
plt.show()


