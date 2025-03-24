from BertRelevanceClassifier import BertRelevanceClassifier, Dataset
import torch

DROPOUT_RATE = 0.1
NUM_LABELS = 3
EMBEDDING_SIZE = 768
TEST_EMBED_PATH = './embeddings/X_test_clean_nohashtag.npy'
TEST_LABELS_PATH = './veriler/y_test_clean_nohashtag.csv'
COMPANIES_EMBED_PATH = './embeddings/companies.npy'

test_data = Dataset(TEST_EMBED_PATH, COMPANIES_EMBED_PATH, TEST_LABELS_PATH)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

model = BertRelevanceClassifier(NUM_LABELS, EMBEDDING_SIZE, DROPOUT_RATE)

for i in range(213):
    model_path = './models/model_' + str(i)
    model.load_state_dict(torch.load(model_path)) 
     
    running_accuracy = 0 
    total = 0 
 
    with torch.no_grad(): 
        for data in test_loader: 
            inputs, outputs = data 
            inputs = inputs.to(torch.float32)
            #outputs = outputs.to(torch.float32) 
            predicted_outputs = model(inputs) 
            _, predicted = torch.max(predicted_outputs, 1) 
            total += outputs.size(0) 
            running_accuracy += (predicted == outputs).sum().item() 
 
        print('Model {}'.format(i))
        print('Accuracy: %.1f %%' % (100 * running_accuracy / total)) 