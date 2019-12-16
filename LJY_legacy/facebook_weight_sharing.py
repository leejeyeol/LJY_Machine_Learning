import torch
import torch.autograd
import torch.nn

def calc_gradient(model, input1, input2):
    x1, x2 = torch.autograd.Variable(input1, requires_grad=True), torch.autograd.Variable(input2, requires_grad=True)
    x1, x2 = x1 ** 2, x2 ** 2

    y1, y2 = model(x1, x2)

    g1, g2 = torch.autograd.grad(y1, x1, retain_graph=True, create_graph=True, only_inputs=True)[0], torch.autograd.grad(y2, x2, retain_graph=True, create_graph=True, only_inputs=True)[0]
    g_p1, g_p2 = g1 ** 2, g2 ** 2

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

    def forward(self, input1, input2):
        input1 = self.linear_1(input1)
        input2 = self.linear_2(input2)
        output1 = self.shared(input1)
        output2 = self.shared(input2)

        return output1, output2

model = TestModel()
for p in model.parameters():
    p.requires_grad = True

model.zero_grad()
k1, k2 = torch.ones(1), torch.ones(1)
k1, k2 = torch.autograd.Variable(k1), torch.autograd.Variable(k2)
j1, j2 = model(k1, k2)
j1, j2 = j1.mean(), j2.mean()

j1.backward()
j2.backward()

g_p1, g_p2 = calc_gradient(model, k1.data, k2.data)

g_p1.backward()
g_p2.backward()
