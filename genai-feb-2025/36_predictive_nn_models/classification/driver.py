from dataloader import *
from model import *
from trainer import *

# get data loaders
data_dir = "C:/Users/pc/Documents/nn/classification"
batch_size = 100
train_loader = get_loader(data_dir, "train.csv", batch_size, True)
test_loader = get_loader(data_dir, "test.csv", batch_size, False)
model = CreditScoreModel2(29, 50, 25, 10, 5, 3)

# prepare train parameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
epochs = 50

# train the model
trainer = Trainer(model)
trainer.train(epochs, optimizer, loss_fn, train_loader)

# print weights
print(model.state_dict())

# infer the model
print(trainer.infer(test_loader, loss_fn))

trainer.save_model(data_dir)
