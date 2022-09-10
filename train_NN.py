import torch
import torch.nn as nn
from tony_dataset import CTGdataset
from neural_net import Neural_Net

'''
STEP 1: LOADING DATASET & MAKING DATASET ITERABLE
'''
batch_size = 50
num_epochs = 10
n_iters = 3000

CTG_dataset = CTGdataset()

train_dataset, test_dataset = torch.utils.data.random_split(CTG_dataset, 
    [int(0.5 * len(CTG_dataset)), len(CTG_dataset) - int(0.5 * len(CTG_dataset))])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)

'''
STEP 2: INSTANTIATE MODEL CLASS
'''
input_dim = 22
hidden_dim = 12
output_dim = 10

model = Neural_Net(input_dim, hidden_dim, output_dim)

'''
STEP 3: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 4: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Consider if Adam or SGD

'''
STEP 5: TRAIN THE MODEL
'''
iter = 0
for epoch in range(num_epochs):
    for i, (samples, labels) in enumerate(train_loader):

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(samples)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for samples, labels in test_loader:

                # Forward pass only to get logits/output
                outputs = model(samples)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
