import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
  
warnings.filterwarnings(action = 'ignore')
  
import gensim
import gensim.downloader
from gensim.models import Word2Vec
  
class embeddedWord:
  def __init__(self, word=None, embedding = None, count=0):
    self.word = word
    self.embedding = embedding
    self.count = count

class document: 
  def __init__(self):
    self.words = {}
    self.tot = 0
  def add_word(self, word):
    if word.word in self.words:
      self.words[word.word].count += word.count
      self.tot+=word.count
    else:
      self.words[word.word] = word
      self.tot+=word.count
  def __str__(self):
    temp = f"total length: {self.tot}\n"
    for i in self.words:
      temp+="word: " + str(self.words[i].word) +", count: "+str(self.words[i].count) + ", vec[0:5]: " + str(self.words[i].embedding[0:5])+"\n"
    return temp
  def concat(self, doc):
    for w in doc.words:
      self.add_word(doc.words[w])
    return self
#  Reads ‘alice.txt’ file
sample = open("text/test.txt", "r")
s = sample.read()
  
# Replaces escape character with space
f = s.replace("\n", " ")
  
data = []
w2vdata = []
w2vCounts = []
doc = document()
doc2 = document()

for i in sent_tokenize(f):
  temp = []    
  # tokenize the sentence into words
  for j in word_tokenize(i):
    temp.append(j.lower())
  
  data.append(temp)
  w2vdata.append({})
  w2vCounts.append({})

print(data)
w2v = gensim.downloader.load('word2vec-google-news-300')

for sent in range(len(data)):
  for i in range(len(data[sent])):
    if data[sent][i] in w2v.index_to_key:
      if sent<3:
        doc.add_word(embeddedWord(word=data[sent][i], embedding=w2v[data[sent][i]], count=1))
      else:
        doc2.add_word(embeddedWord(word=data[sent][i], embedding=w2v[data[sent][i]], count=1))
      w2vdata[sent][data[sent][i]] = w2v[data[sent][i]]
      w2vCounts[sent][data[sent][i]] = w2vCounts[sent].get(data[sent][i],0) + 1
      #print(w2v[data[0][i]])
print(doc)
print(doc2)
print(doc.concat(doc2))