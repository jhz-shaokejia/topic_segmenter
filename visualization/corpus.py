
import warnings
from functools import wraps
from collections import defaultdict

import re
import cPickle as pk
import pandas as pd

from nltk.corpus import stopwords as NLTKstopwords
from gensim import corpora



def load_corpus(pickle_file):
    """ Returns the Corpus instance """
    with open(pickle_file, 'rb') as f:
        return pk.load(f)



class Corpus(object):
    """ Corpus object, to be used for Models such as TFIDF and word2vec / GloVe """

    def __init__(self, stopwords='NLTK', punctuation='.,!`', user_string='<u\w*>'):
        """ Note - Pass an iterable to stopwords to override NLTKs stopwords """
        super(Corpus, self).__init__()
        self.STOPWORDS = stopwords
        self.PUNCTUATION = punctuation
        self.USER_STRING = user_string

        # TODO - corpus storage

        # Initialize to None
        self.documents = None
        self.dictionary = None
        self.corpus = None


    def check_init(func):
        """ Wraps function with a initialized documents checker """
        @wraps(func)
        def _func(*args):
            if self.documents is not None:
                warnings.warn('Corpus must be processed first')
                # return None
            else:
                return func(*args)

        return _func


    ### PROPERTIES -----------------------------------

    @property
    def STOPWORDS(self):
        """ Set of stopwords. These words are not incorporated into the corpus dictionary """
        return self.__STOPWORDS

    @STOPWORDS.setter
    def STOPWORDS(self, stopwords):
        if stopwords != 'NLTK':
            self.__STOPWORDS = set(NLTKstopwords.words('english'))  # from NLTK
        else:
            # Override NLTKs stopword list
            self.__STOPWORDS = set(stopwords)


    @property
    def PUNCTUATION(self):
        """ Set of punctuation symbols to be removed. These symbols are removed from the documents """
        return self.__PUNCTUATION


    @property
    def USER_STRING(self):
        """ Set of punctuation symbols to be removed. These symbols are removed from the documents """
        return self.__USER_STRING

    @USER_STRING.getter
    def USER_STRING(self):
        return self.__USER_STRING.pattern  # resolves to the pattern compiled, not the compilation

    @USER_STRING.setter
    def USER_STRING(self, user_string):
        if isinstance(user_string, str):  # check if it is not precompiled
            self.__USER_STRING = re.compile(user_string)
        else:
            self.__USER_STRING = user_string


    ### LOADERS -----------------------------------

    def from_topic_table(self, topics_table):
        """ Generate the set of documents from the topic_table list """
        grouped = TOPICS_DATA[['topic', 'text']].groupby('topic')
        docs = grouped.agg(lambda x: ' '.join( map(lambda x: x.encode('ascii', 'replace'), x) ))

        self.documents = docs.values.flatten()


    def from_topic_table_csv(self, path_to_table):
        """ Reads the csv file and then calls `from_topics_table` """
        topics_table = pd.read_csv(path_to_table, encoding='UTF-8', index_col=0)
        self.from_topic_table(topics_table)


    ### PROCESSORS -----------------------------------

    def remove_punct(self, doc):
        """ Removes each of the symbols specified in the PUNCTUTATION property """
        return reduce( lambda d, s: d.replace(s, ''), [doc,] + list(self.PUNCTUATION) )


    def remove_usernames(self, doc):
        """ Removes the usernames from a document, according to the specified in the PUNCTUTATION property """
        return self.__USER_STRING.sub('', doc)


    def get_full_remover(self):
        """ Returns the current punctuation + username remover """
        def remover(document):
            """ Removes punctuation marks from documents """
            return reduce( lambda d, f: f(d), [document, self.remove_punct, self.remove_usernames] )
        return remover


    @check_init
    def process(self):
        docs = reduce( lambda x, f: map(f, x), [self.documents, self.remove_punct, self.remove_usernames] )

        # Remove stopwords and tokenize
        texts = map( lambda doc: [word for word in doc.lower().split() if word not in STOPWORDS], docs )

        # Generate a dictionary with the term frequency
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        # remove words that appear only once
        texts = map( lambda text: filter(lambda token: frequency[token] > 1, text), texts )

        # Generate dictionary of terms
        self.dictionary = corpora.Dictionary(texts)
        # dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference

        # Finally, generate the corpus
        self.corpus = map( self.dictionary.doc2bow, texts )


    def store_corpus(self, verbose=False):
        """ Stores the corpus as a pickle file """
        with open(self.pickle_path, 'wb') as f:
            pk.dump(f, self)

        if verbose:
            print(' -- Saved pickle file to: {}'.fortmat(pickle_path))
