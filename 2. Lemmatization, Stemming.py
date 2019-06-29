from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

PHRASE = 'Cats pants and wolves'

tokenizer = TreebankWordTokenizer()
porter = PorterStemmer()
word_net = WordNetLemmatizer()

tokens = tokenizer.tokenize(PHRASE)
print(tokens)

print("Porter   : ", ' '.join(porter.stem(token) for token in tokens))
print("Word Net : ", ' '.join(word_net.lemmatize(token) for token in tokens))
