import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

m = XOR()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(m.parameters(), lr=1e-3)

epochs = 3000
minibatch_size = 32

pairs = [(np.asarray([0.0,0.0]), [0]),
         (np.asarray([0.0,1.0]), [1]),
         (np.asarray([1.0,0.0]), [1]),
         (np.asarray([1.0,1.0]), [0])]

X = torch.Tensor([x[0] for x in pairs])
Y = torch.Tensor([x[1] for x in pairs])

for i in range(epochs):
        
    for batch in range(4):
        # forward pass
        y_pred = m(X)
        
        # compute and print loss
        loss = loss_fn(y_pred, Y)
        #print(i, batch, loss.data[0])

        # reset gradients
        optimizer.zero_grad()
        
        # backwards pass
        loss.backward()
        
        # step the optimizer - update the weights
        optimizer.step()

print("Predict:")
print("f(1,1) = {}".format(int(m(torch.Tensor([1.0,0])))))
