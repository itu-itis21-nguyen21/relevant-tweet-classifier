import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr', device='cuda')

# Read data
irrelevant_file_path = './irrelevant_clean_nohashtag_newfeatures2_6.csv'
irrelevant_df = pd.read_csv(irrelevant_file_path, sep=',')
sasa_file_path = './sasa_clean_nohashtag_newfeatures2_6.csv'
sasa_df = pd.read_csv(sasa_file_path, sep=',')
erdem_file_path = './erdem_clean_nohashtag_newfeatures2_6.csv'
erdem_df = pd.read_csv(erdem_file_path, sep=',')
frames = [sasa_df, erdem_df, irrelevant_df]
df = pd.concat(frames)

X_train = df['tweet_text']
y_train = df['relevance']

y_train = y_train.to_frame()
y_train.to_csv('./y_train_clean_nohashtag.csv', index=False)

X_train = X_train.tolist()
X_train_embeds = model.encode(X_train)
#print(X_train_embeds.shape)
np.save('./X_train_clean_nohashtag.npy', X_train_embeds)