import torch
import pandas as pd
import numpy as np

HIDDEN_SIZE = 128

class Dataset:
    def __init__(self, tweet_embed_file_path, companies_embed_file_path, label_file_path):
        self.data = np.load(tweet_embed_file_path)
        #self.data = np.squeeze(self.data, axis=1)       # if needed
        #self.companies = np.load(companies_embed_file_path)
        #self.companies = np.tile(self.companies.T, (self.tweets.shape[0], 1))
        #self.data = np.concatenate((self.tweets, self.companies), axis=1)
        print(self.data.shape)
        self.labels = pd.read_csv(label_file_path, sep=',')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        embedding = self.data[idx]
        label = self.labels['relevance'][idx]
        return embedding, label

# Step 2: Define the finetuning network (fully connected network with dropout)
class BertRelevanceClassifier(torch.nn.Module):
    def __init__(self, num_labels, embedding_size, dropout_rate):
        super(BertRelevanceClassifier, self).__init__()
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        # Fully connected layer with dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        #self.out = torch.nn.Linear(self.embedding_size, 1)      # binary classification
        #self.out = torch.nn.Linear(self.embedding_size, self.num_labels)
        self.linear1 = torch.nn.Linear(self.embedding_size, HIDDEN_SIZE)
        self.nonlinear = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = torch.nn.Linear(HIDDEN_SIZE, self.num_labels)

    def forward(self, embeddings):
        #Takes a batch of tweet embeddings and returns logits for the relevant classes.
        out = self.nonlinear(self.linear1(embeddings))
        out = self.dropout(out)
        out = self.nonlinear(self.linear2(out))
        out = self.dropout(out)
        return self.linear3(out)


