from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns

REVIEWS = [
    'great movie',
    'not a great movie',
    'did not like it',
    'i like it',
    'nice one'
]

tfidf = TfidfVectorizer(min_df=2,
                        max_df=0.5,
                        ngram_range=(1, 2))

features = tfidf.fit_transform(REVIEWS)

print(pd.DataFrame(data=features.todense(),
                   columns=tfidf.get_feature_names()))