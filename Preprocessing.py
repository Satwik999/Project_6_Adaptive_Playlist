import spacy
import re
import numpy as np
import pandas as pd
from collections import Counter

nlp = spacy.load('en_core_web_sm')

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS



def cleaner(df1):
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    pattern = re.compile(r"[A-Za-z\-]{3,50}")
    df1['Title'] = df1['Title'].apply(lambda x: ' '.join(re.findall(pattern,str(x))))
    df1['Description'] = df1['Description'].apply(lambda x: ' '.join(re.findall(pattern,str(x))))
    df1['Tags'] = df1['Tags'].apply(lambda x: ' '.join(re.findall(pattern,str(x))))
    # Removing numbers & Extras
    df1['Title'] = df1['Title'].apply(lambda x: str(x).replace('\d+', ''))
    df1['Description'] = df1['Description'].apply(lambda x: str(x).replace('\d+', ''))
    df1['Tags'] = df1['Tags'].apply(lambda x: str(x).replace('\d+', ''))
    df1['Description'] = df1['Description'].apply(lambda x: str(x).replace('http\S+|www.\S+|html\S+|-\S+|com\S+|youtube\S+|https\S+',''))
    df1['Tags'] = df1['Tags'].apply(lambda x: str(x).replace('http\S+|www.\S+|html\S+|-\S+|com\S+|youtube\S+|https\S+',''))
    df1.drop(['VideoID'],axis=1,inplace=True)
    return df1


#To only allow valid tokens which are not stop words and punctuation symbols.
def is_token_allowed(token):
    if (not token or not token.string.strip() or
        token.is_stop or token.is_punct):
        return False
    return True


def preprocess_token(token):
    # Reduce token to its lowercase lemma form
    token = token.lemma_.strip().lower()
#     token = token.replace('http\S+|www.\S+|html\S+|-\S+|com\S+|youtube\S+|https\S+','')#, case=False)
    return token

def get_tokens(main_df):
    cleaned = cleaner(main_df)
    nlp_df= pd.DataFrame(columns=["Title_token", "Desc_token", "Tags_token"])

    for i in cleaned.index:
        a = {}
        row = cleaned.iloc[i,].tolist()
        for j in range(len(row)):
            doc_item = nlp(row[j])
            tokens = [preprocess_token(token) for token in doc_item if is_token_allowed(token)]
            a.update({nlp_df.columns[j]:' '.join(tokens)})
        nlp_df = nlp_df.append(a, ignore_index=True)
        
        test = nlp_df.apply(lambda x:' '.join(x.dropna().astype(str)) ,axis=1)
#         break
    return test
# ' '.join(str(x.dropna()).strip("[]"))   [nlp_df.columns[0:3]]     str(tokens).strip("[]")
