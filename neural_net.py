import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tony_dataset import CTGdataset

class Neural_Net(nn.Module):
    '''
    An untrained neural network
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Neural_Net, self).__init__()
        # Linear function 1
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.sigmoid1 = nn.Sigmoid()

        # Linear function 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.sigmoid2 = nn.Sigmoid()

        # Linear function 3 (readout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.sigmoid1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.sigmoid2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)
        return out

# Train the neural network with the given hyperparameters
def train_nn(model, train_dataset, test_dataset, num_epochs:int = 60, 
    batch_size:int = 100, l_r = 0.001, optimizerName:str = "Adam"):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    assert optimizerName in ["Adam", "SGD"]
    if optimizerName == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr= l_r)
    else: # optimizer = SGD
        optimizer = torch.optim.SGD(model.parameters(), lr= l_r)
    criterion = nn.CrossEntropyLoss() # Instantiate loss class

    # train the model

    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Calculate performance metrics at the end of each epoch      
        print(f"[INFO]: Epoch {epoch+1} of {num_epochs}")
        correct = 0
        total = 0
        # Iterate through training dataset, calculate training accuracy
        for samples, labels in train_loader:
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        train_accuracy = 100 * correct / total

        correct = 0
        total = 0
        # Iterate through test dataset, calculate testing accuracy
        for samples, labels in test_loader:
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_accuracy = 100 * correct / total

        # Print Loss
        print('Loss: {}, Training Accuracy: {}, Test Accuracy: {}.'.format(loss.item(), train_accuracy, test_accuracy))


if __name__ == '__main__':
    input_dim = 8
    hidden_dim = 5
    output_dim = 3
    data_count = 200

    model = Neural_Net(input_dim, hidden_dim, output_dim)
    torch.save(model,f = 'model.pt')
    model = torch.load("model.pt")

    model.eval()
    # x = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.])
    x = torch.normal(mean = torch.tensor([[0. for i in range(input_dim)] for j in range(data_count)]), std = 1.)
    print("x: ", x)
    outputs = model(x)  # Forward pass to get logits
    _, predicted = torch.max(outputs.data, dim = 1)   # Get predictions from the max value
    
    # print("outputs: ", outputs)
    print(predicted)