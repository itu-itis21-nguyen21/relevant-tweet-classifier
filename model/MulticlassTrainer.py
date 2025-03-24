from BertRelevanceClassifier import BertRelevanceClassifier, Dataset
import torch

DROPOUT_RATE = 0.1
NUM_LABELS = 3
EMBEDDING_SIZE = 768
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 200
TRAIN_EMBED_PATH = './embeddings/X_train_clean_nohashtag.npy'
TRAIN_LABELS_PATH = './veriler/y_train_clean_nohashtag.csv'
COMPANIES_EMBED_PATH = './embeddings/companies.npy'

train_data = Dataset(TRAIN_EMBED_PATH, COMPANIES_EMBED_PATH, TRAIN_LABELS_PATH)
training_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

model = BertRelevanceClassifier(NUM_LABELS, EMBEDDING_SIZE, DROPOUT_RATE)

# Step 3: Define train function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(torch.float32)
        #labels = labels.unsqueeze(1).float()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

epoch_number = 0
best_loss = 1_000_000.
model_count = 0
training_losses = []

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    training_losses.append(avg_loss)

    # Track best performance, and save the model's state
    if avg_loss < best_loss:
        best_loss = avg_loss
        model_path = './models/model_{}'.format(model_count)
        torch.save(model.state_dict(), model_path)
        model_count += 1

    epoch_number += 1