import itertools as it
import math

# following changes are for compatibilities
UseNLTK = False

if UseNLTK:
	from nltk.tokenize import word_tokenize
	from nltk.stem import PorterStemmer
else:
	class PorterStemmer:
		def stem(self, _word: str):
			return _word

	import re
	_SepExp = re.compile("[^a-zA-Z0-9]+")

	def word_tokenize(_data):
		return re.split(_SepExp, _data)

class BM25:
	def __init__(self):
		self.stemmer = PorterStemmer()

	def normalizeToken(self, _word):
		return self.stemmer.stem(_word).lower() if len(_word) > 3 else (_word if _word.isupper() else _word.lower())

	@staticmethod
	def _stringMatch(_query, _document):
		qlen_ = len(_query)
		qset_ = set(_query)
		def exactMatch(_pos):
			return all((_document[_pos + i] == q for i, q in enumerate(_query)))
		tokenMatches_ = fullMatches_ = matches_ = 0
		for i, t in enumerate(_document):
			if t in qset_:
				tokenMatches_ += 1
				matches_ += 1
			else:
				matches_ = 0
			if matches_ >= qlen_ and exactMatch(i - qlen_ + 1):
				fullMatches_ += 1
		return fullMatches_, tokenMatches_

	def calculate(self, _documents, _queries, _termBias = None):
		_documents = list(map(lambda d : list(map(self.normalizeToken, word_tokenize(d))), _documents))
		_queries = list(map(lambda q: list(map(self.normalizeToken, word_tokenize(q))), _queries))
		termScores_ = self._calculate(_documents, _queries)
		_queries = list(map(lambda x: [x], set(it.chain.from_iterable(_queries))))
		tokenScores_ = self._calculate(_documents, _queries)
		if _termBias is None:
			return list(zip(termScores_, tokenScores_))
		else:
			return list(map(lambda x: x[0] * _termBias + x[1] * (1 - _termBias),zip(termScores_, tokenScores_)))


	def _calculate(self, _documents, _queries):
		documentScores_ = [0] * len(_documents)
		totalIdf_ = 0
		for query_ in _queries:
			documentScore_ = [0] * len(_documents)
			docCount_ = 0
			for i, doc_ in enumerate(_documents):
				termCount_, _ = BM25._stringMatch(query_, doc_)
				if termCount_:
					tf_ = termCount_ * len(query_) / len(doc_)
					documentScores_[i] = tf_ * 2.5 / (tf_ + 1.5)
					docCount_ += 1
			idf_ = math.log((len(_documents) - termCount_ + 0.5) / (termCount_ + 0.5) + 1)
			totalIdf_ += idf_
			for i, s in enumerate(documentScore_):
				documentScores_[i] += s * idf_
		for i in range(len(documentScores_)):
			documentScores_[i] /= totalIdf_
		return documentScores_

def unittest():
	scorer_ = BM25()
	queries_ = ["NLTK", "Natural Language Toolkit"]
	documents_ = [
		"NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.",
		"Natural Language Processing (NLP) is a process of manipulating or understanding the text or speech by any software or machine. An analogy is that humans interact and understand each other’s views and respond with the appropriate answer. In NLP, this interaction, understanding, and response are made by a computer instead of a human.",
		"NLTK -- the Natural Language Toolkit -- is a suite of open source Python modules, data sets, and tutorials supporting research and development in Natural Language Processing. NLTK requires Python version 3.6, 3.7, 3.8, or 3.9."
	]
	print(scorer_.calculate(documents_, queries_))
	print(scorer_.calculate(documents_, queries_, 0.7))

if __name__ == "__main__":
	unittest()
