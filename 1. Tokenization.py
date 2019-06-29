import nltk
from nltk.tokenize import WhitespaceTokenizer, TreebankWordTokenizer, WordPunctTokenizer

PHRASE = 'I hadn\'t taken my breakfast before I came to Sharan\'s class'

white_space = WhitespaceTokenizer()
tree_bank_word = TreebankWordTokenizer()
word_punct = WordPunctTokenizer()

print("WhitespaceTokenizer   : ", white_space.tokenize(PHRASE))
print("TreebankWordTokenizer : ", tree_bank_word.tokenize(PHRASE))
print("WordPunctTokenizer    : ", word_punct.tokenize(PHRASE))
