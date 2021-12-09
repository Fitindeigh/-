
import torch
import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)


x_train = torch.rand(100)    
x_train = x_train * 20 - 10      # else x_train > 0

y_train = x_train**2 - 4      
 

# plt.plot(x_train.numpy(), y_train.numpy(), 'o')  
# plt.title('$y = x^2 - 4$');


noise = torch.randn(y_train.shape) / 0.2        #create noise

# plt.plot(x_train.numpy(), noise.numpy(), 'o') 
# plt.axis([-1.1, 1.1, -4.2, -1.8])
# plt.title('Gaussian noise');



y_train = y_train + noise

# plt.plot(x_train.numpy(), y_train.numpy(), 'o')
# plt.title('noisy x^2 - 4')
# plt.xlabel('x_train')
# plt.ylabel('y_train');


x_train.unsqueeze_(1)         # create 2d tensor
y_train.unsqueeze_(1)


x_validation = torch.linspace(-10, 10, 100)
y_validation = (x_validation.data)**2 - 4
# plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
# plt.title('x^2 - 4')
# plt.xlabel('x_validation')
# plt.ylabel('y_validation');

x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)


class XSqrMinus4(torch.nn.Module):             # Initialization NN
    def __init__(self, n_hidden_neurons):
        super(XSqrMinus4, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=1, out_features=n_hidden_neurons, bias=True)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(in_features=n_hidden_neurons, out_features=n_hidden_neurons, bias=True)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(in_features=n_hidden_neurons, out_features=1, bias=True)



    def forward(self, x):               #forward pass
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)

        return x

XSqr_Minus4 = XSqrMinus4(20)

def predict(Sqr, x, y):             
    y_pred = Sqr.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction');
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')


optimizer = torch.optim.Adam(XSqr_Minus4.parameters(), lr=0.01)

# def loss(pred, target):
#     BCE = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
#     return BCE

def loss(pred, target):
    MSE = (pred - target) ** 2
    return MSE.mean()

for Loss_plot in range(10):
    for epoch_index in range(200):
        optimizer.zero_grad()

        y_pred = XSqr_Minus4.forward(x_train)
        loss_val = loss(y_pred, y_train)

        loss_val.backward()

        optimizer.step()
    print(loss(y_pred, y_train))

predict(XSqr_Minus4, x_validation, y_validation)

