import torch
import torch.autograd
import torch.optim as optim

def calc_gradient(model, input1, input2):

    x1 = input1 **2
    x2 = input2 **2
    x1, x2 = torch.autograd.Variable(x1, requires_grad=True), torch.autograd.Variable(x2, requires_grad=True)

    y1, y2 = model(x1, x2)

    g1, g2 = torch.autograd.grad(y1, x1, retain_graph=True, create_graph=True, only_inputs=True)[0], torch.autograd.grad(y2, x2, retain_graph=True, create_graph=True, only_inputs=True)[0]
    g_p1, g_p2 = (g1 ** 2).view(1), (g2 ** 2).view(1)

    return g_p1, g_p2


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(1, 1)
        )
        self.linear_2 = torch.nn.Sequential(
            torch.nn.Linear(1, 1)
        )
        self.shared = torch.nn.Sequential(
        torch.nn.Linear(1, 1)
        )
        self.linear_1_2 = torch.nn.Sequential(
            torch.nn.Linear(1, 1)
        )
        self.linear_2_2 = torch.nn.Sequential(
            torch.nn.Linear(1, 1)
        )

    def forward(self, input1, input2):
        input1 = self.linear_1(input1)
        input2 = self.linear_2(input2)
        input1 = self.shared(input1)
        input2 = self.shared(input2)
        output1 = self.linear_1_2(input1)
        output2 = self.linear_2_2(input2)

        return output1, output2
model = TestModel()
optimizer = optim.SGD(model.parameters(), lr=0.0005)

for p in model.parameters():
    p.requires_grad = True

for i in range(10):
    model.zero_grad()
    k01, k02 = torch.ones(1), torch.ones(1)
    k1, k2 = torch.autograd.Variable(k01), torch.autograd.Variable(k02)
    j1, j2 = model(k1, k2)
    j1, j2 = j1.mean(), j2.mean()

    j1.backward()
    j2.backward()

    g_p1, g_p2 = calc_gradient(model, k1.data, k2.data)

    g_p1.backward()
    g_p2.backward()

    D = g_p1 + g_p2
    optimizer.step()


    print(i)

for p in model.parameters():
    p.requires_grad = False

