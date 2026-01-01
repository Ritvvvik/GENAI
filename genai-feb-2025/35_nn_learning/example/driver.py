from dataloader import *
from model import *
from trainer import *

# get data loaders
data_dir = "C:/Users/pc/Documents/nn/non-linear"
batch_size = 10
train_loader = get_loader(data_dir, "train.csv", batch_size, True)
test_loader = get_loader(data_dir, "test.csv", batch_size, False)
model = RegressionModel3(1, 5, 1)

# prepare train parameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
epochs = 300

# train the model
trainer = Trainer(model)
trainer.train(epochs, optimizer, loss_fn, train_loader)

# print weights
print(model.state_dict())

# infer the model
print(trainer.infer(test_loader, loss_fn))
