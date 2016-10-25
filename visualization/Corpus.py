
import warnings
from functools import wraps
import re

from nltk.corpus import stopwords as NLTKstopwords

from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora


class Corpus(object):
    """ Corpus object, to be used for Models such as TFIDF and word2vec / GloVe """

    def __init__(self, stopwords='NLTK', punctuation='.,!`', user_string='<u\w*>'):
        """ Note - Pass an iterable to stopwords to override NLTKs stopwords """
        super(Corpus, self).__init__()
        self.documents = None
        self.STOPWORDS = stopwords
        self.PUNCTUATION = punctuation
        self.USER_STRING = user_string


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

    def get_remover(self):
        def remover(document):
            """ Reduce  """
            return reduce( lambda d, s: d.replace(s, ''), [document,] + self.STOPWORDS + list(self.PUNCTUATION) )

    @check_init
    def remove_punct_stops(self, corpus):
        # Removes each of the symbols
        return map(self.get_remover(), self.documents)


    @check_init
    def remove_usernames(self, doc):
        return self.__USER_STRING.sub('', doc)


    def process(self):
        pass
