import pandas as pd
import numpy as np
import math
import ttp
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#SASA_MAX = 6
#ERDEM_MAX = 3

tr_stopwords = stopwords.words('turkish')

def add_erdemoglu(tweet):
    tokens = ['erdemoglu', 'erdemoglutr']
    token_rate = [0.966, 0.034]
    chosen_token = np.random.choice(tokens, p=token_rate)
    added_position = np.random.choice(len(tweet))
    tweet.insert(added_position, chosen_token)
    return tweet

# Step 1: Read the CSV file using pandas and create predictors and labels
sasa_file_path = './veriler/all_sasa_cleaned.csv'
sasa_df = pd.read_csv(sasa_file_path, sep=',')
erdemoglu_file_path = './veriler/all_holding_cleaned.csv'
erdemoglu_df = pd.read_csv(erdemoglu_file_path, sep=',')
erdemoglu_df['relevance'] = erdemoglu_df['relevance'].replace(1, 2)
#print(erdemoglu_df.head(10))
#print(sasa_df.shape)
#print(erdemoglu_df.shape)

frames = [sasa_df, erdemoglu_df]
df = pd.concat(frames)
#print(df.shape)

def preprocess(df):
    # Remove links
    df['tweet_text'] = df['tweet_text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.remove_emoji(x))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.remove_emoticon(x))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.remove_hashtag_and_word(x))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.lower(x))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.vanish_punc(x))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.remove_newline_char(x))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.dup_vanish(x))
    #df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.preprocess_sentence(x, tr_stopwords))       # this line removes all numbers, please check later
    # Remove superscripts
    df['tweet_text'] = df['tweet_text'].replace(r'\u2074+', '', regex=True).replace(r'\u2070+', '', regex=True)
    # Change all Turkish characters to English
    translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
    df['tweet_text'] = df['tweet_text'].apply(lambda x: x.translate(translationTable))
    # Remove double quotes
    #df['tweet_text'] = df['tweet_text'].replace('"', '', regex=True)
    # Remove 1-character words
    df['tweet_text'] = df['tweet_text'].apply(lambda i: ' '.join(filter(lambda j: len(j) > 1, i.split())))
    # Remove all sasa and erdemoglu tokens
    #df['tweet_text'] = df['tweet_text'].replace(r'sasa\S*', '', regex=True).replace(r'erdemoglu\S*', '', regex=True)
    # Remove extra whitespaces
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ttp.remove_extra_spaces(x))
    #print(df['tweet_text'].head(10))
    # Count client mentions and normalize
    df['sasa_mentions'] = df['tweet_text'].apply(lambda x: x.count('sasa'))
    df['erdem_mentions'] = df['tweet_text'].apply(lambda x: x.count('erdemoglu'))
    df['sasa_mentions'] = df['sasa_mentions'].apply(lambda x: 0 if x == 0 else math.log(x))
    df['erdem_mentions'] = df['erdem_mentions'].apply(lambda x: 0 if x == 0 else math.log(x))
    return df

df = preprocess(df)

# Insert erdemoglu tokens to irrelevant tweets to train them with all classes
irrelevant_df = df.loc[df['relevance'] == 0]
#irrelevant_df['tweet_text'] = irrelevant_df['tweet_text'].apply(lambda x: x.split())
#irrelevant_df['tweet_text'] = irrelevant_df['tweet_text'].apply(lambda x: add_erdemoglu(x))
#irrelevant_df['tweet_text'] = irrelevant_df['tweet_text'].apply(lambda x: ' '.join(x))
#print(irrelevant_df.head(10))

# Split train and test sets
sasa_df = df.loc[df['relevance'] == 1]
erdem_df = df.loc[df['relevance'] == 2]
train_irrelevant, test_irrelevant = train_test_split(irrelevant_df, test_size=0.2, random_state=42)
train_sasa, test_sasa = train_test_split(sasa_df, test_size=0.2, random_state=42)
train_erdem, test_erdem = train_test_split(erdem_df, test_size=0.2, random_state=42)

# Save new datasets
train_irrelevant.to_csv('./veriler/train/irrelevant_clean_nohashtag.csv', index=False)
test_irrelevant.to_csv('./veriler/test/irrelevant_clean_nohashtag.csv', index=False)
train_sasa.to_csv('./veriler/train/sasa_clean_nohashtag.csv', index=False)
test_sasa.to_csv('./veriler/test/sasa_clean_nohashtag.csv', index=False)
train_erdem.to_csv('./veriler/train/erdem_clean_nohashtag.csv', index=False)
test_erdem.to_csv('./veriler/test/erdem_clean_nohashtag.csv', index=False)

