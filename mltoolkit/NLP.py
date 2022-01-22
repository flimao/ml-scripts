#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import re
from six import string_types
from unidecode import unidecode
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.models import doc2vec

def remove_links(string):
    s = re.sub(r'http\S+', '', string, flags=re.MULTILINE)
    return s

def remove_hashtags(string):
    s = re.sub(r'#(\w+)', '', string, flags=re.MULTILINE)
    return s

def remove_mentions(string):
    s = re.sub(r'@(\w+)', '', string, flags=re.MULTILINE)
    return s

def remove_numbers(string):
    s = re.sub(r'\d', '', string)

    return s

def remove_punkt(string):
    s = re.sub(r'\W', ' ', string)
    return s

def remove_special_caract(string):
    s = unidecode(string)
    return s

def lowercase(string):
    s = string.lower()
    return s

def tokenize(string, tokenizer):
    tokens = tokenizer(string)

    return tokens

def remove_stopwords(word_list, stopword_list):
    filtered_words = []
    for w in word_list:
        if w not in stopword_list:
            filtered_words.append(w)
    
    return filtered_words

# get radicals via stemmer or lemmatizer
def get_radicals(word_list, radicalizer):
    radicalized_words = []
    for w in word_list:
        r_words = radicalizer(w)
        radicalized_words.append(r_words)
    
    return radicalized_words

# para o spacy, as funções são aplicadas em uma ordem e de forma diferentes.
# criamos uma função específica para análise com spacy
def tokenize_remove_stopwords_get_radicals_spacy(word_list, nlp, stopword_list = None, retornar_string = True):
    
    if stopword_list is not None:
        for stopword in stopword_list:
            nlp.vocab[stopword].is_stop = True
    
    if isinstance(word_list, str):
        tokens = nlp(word_list)
    else:
        tokens = nlp(' '.join(word_list))
    
    radicalized_words = [ 
        token.lemma_ 
        for token in tokens 
        if not token.is_stop and token.lemma_.strip() != '' and re.sub(r'\W', '', token.lemma_) != ''
    ]
    if retornar_string:
        return ' '.join(radicalized_words)
    
    else:
        return radicalized_words


# build pipe and push string through
def preprocessing(string, preproc_funs_args):
    # 'preproc_funs_args' is a list. it may contain functions or tuples containing a function
    # and a dict of arguments
    
    input_arg = string
    output = None

    for preproc_fun_args in preproc_funs_args:
        if isinstance(preproc_fun_args, tuple):
            preproc_fun, kwargs = preproc_fun_args
        else:
            preproc_fun = preproc_fun_args
            kwargs = dict()

        output = preproc_fun(input_arg, **kwargs)
        input_arg = output
    
    return output

# given a word2vec model and a token-formatted phrase list (list of lists), build assoc vectors
def build_word2vec_vectors(model, phrases, vector_combination):

    X = []
    vector_size = model.vector_size

    for phrase in phrases:

        ntokens = len(phrase)
        vectors = np.zeros(shape = (ntokens, vector_size))

        for i, token in enumerate(phrase):
            try:

                vectors[i, :] = model.wv[token]
            except KeyError:  # token not present in corpus
                vectors[i, :] = 0

        X.append(vector_combination(vectors))
    
    return np.asarray(X)

# função para, dados conjuntos de frases de treino e teste, construir os vetores
# associados
def build_word2vec_model(
    X_train, X_test, 
    vector_combination,
    is_token = False,
    **kwargs
):
    # kwargs = arguments for Word2Vec class
    
    if is_token:
        X_train_tokens = X_train
        X_test_tokens = X_test
    else:
        X_train_tokens = X_train.str.split(' ').to_list()
        X_test_tokens = X_test.str.split(' ').to_list()
    
    # instantiate, build and train model
    w2v_model = models.Word2Vec(
        sentences = X_train_tokens, 
        **kwargs
    )

    # build vectors
    X_train_w2v = build_word2vec_vectors(
        model = w2v_model, 
        phrases = X_train_tokens, 
        vector_combination = vector_combination
    )

    X_test_w2v = build_word2vec_vectors(
        model = w2v_model, 
        phrases = X_test_tokens, 
        vector_combination = vector_combination
    )

    return w2v_model, X_train_w2v, X_test_w2v

# sklearn transformer for word2vec model
class W2VTransformer(TransformerMixin, BaseEstimator):
    """Base Word2Vec module, wraps :class:`~gensim.models.word2vec.Word2Vec`.
    For more information please have a look to `Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: "Efficient
    Estimation of Word Representations in Vector Space" <https://arxiv.org/abs/1301.3781>`_.
    """
    def __init__(self, vector_size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000, vector_combination = lambda x: np.sum(x, axis = 0)):
        """
        Parameters
        ----------
        size : int
            Dimensionality of the feature vectors.
        alpha : float
            The initial learning rate.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sg : int {1, 0}
            Defines the training algorithm. If 1, CBOW is used, otherwise, skip-gram is employed.
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        cbow_mean : int {1,0}
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : callable (object -> int), optional
            A hashing function. Used to create an initial random reproducible vector by hashing the random seed.
        iter : int
            Number of iterations (epochs) over the corpus.
        null_word : int {1, 0}
            If 1, a null pseudo-word will be created for padding when using concatenative L1 (run-of-words)
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        sorted_vocab : int {1,0}
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
        batch_words : int
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        """
        self.gensim_model = None
        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.epochs = epochs
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.vector_combination = vector_combination

    def fit(self, X, y=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : iterable of iterables of str
            The input corpus. X can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        Returns
        -------
        :class:`~gensim.sklearn_api.w2vmodel.W2VTransformer`
            The trained model.
        """

        # X_tokens = X.str.split(' ').to_list()
        X_np = np.array(X, dtype = 'str')
        X_tokens = np.char.split(X_np, ' ')

        self.gensim_model = models.Word2Vec(
            sentences=X_tokens, vector_size=self.vector_size, alpha=self.alpha,
            window=self.window, min_count=self.min_count, max_vocab_size=self.max_vocab_size,
            sample=self.sample, seed=self.seed, workers=self.workers, min_alpha=self.min_alpha,
            sg=self.sg, hs=self.hs, negative=self.negative, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn, epochs=self.epochs, null_word=self.null_word, trim_rule=self.trim_rule,
            sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )
        return self

    def transform(self, words):
        """Get the word vectors the input words.
        Parameters
        ----------
        words : {iterable of str, str}
            Word or a collection of words to be transformed.
        Returns
        -------
        np.ndarray of shape [`len(words)`, `size`]
            A 2D array where each row is the vector of one word.
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        #phrases = words.str.split(' ').to_list()
        words_np = np.array(words, dtype = 'str')
        phrases = np.char.split(words_np, ' ')

        wvs = build_word2vec_vectors(
            model = self.gensim_model,
            phrases = phrases,
            vector_combination = self.vector_combination
        )

        return wvs

    def partial_fit(self, X):
        raise NotImplementedError(
            "'partial_fit' has not been implemented for W2VTransformer. "
            "However, the model can be updated with a fixed vocabulary using Gensim API call."
        )

# function to read corpus and tag sentences in a doc2vec model
def read_corpus(list_sentences, tokens_only = False):
    if tokens_only:
        return list_sentences
    else:
        # For training data, add tags
        lista = []
        for i, line in enumerate(list_sentences):
            lista.append(models.doc2vec.TaggedDocument(line, [i]))

        return lista

# sklearn transformer for doc2vec model
class D2VTransformer(TransformerMixin, BaseEstimator):
    """Base Doc2Vec module, wraps :class:`~gensim.models.doc2vec.Doc2Vec`.
    This model based on `Quoc Le, Tomas Mikolov: "Distributed Representations of Sentences and Documents"
    <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_.
    """
    def __init__(self, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
                 comment=None, trim_rule=None, vector_size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001, hs=0, negative=5, cbow_mean=1,
                 hashfxn=hash, epochs=5, sorted_vocab=1, batch_words=10000):
        """
        Parameters
        ----------
        dm_mean : int {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean. Only applies when `dm_concat=0`.
        dm : int {1,0}, optional
            Defines the training algorithm. If `dm=1` - distributed memory (PV-DM) is used.
            Otherwise, distributed bag of words (PV-DBOW) is employed.
        dbow_words : int {1,0}, optional
            If set to 1 - trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training, If 0, only trains doc-vectors (faster).
        dm_concat : int {1,0}, optional
            If 1, use concatenation of context vectors rather than sum/average.
            Note concatenation results in a much-larger model, as the input is no longer the size of one
            (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words
            in the context strung together.
        dm_tag_count : int, optional
            Expected constant number of document tags per document, when using dm_concat mode.
        docvecs : :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
            A mapping from a string or int tag to its vector representation.
            Either this or `docvecs_mapfile` **MUST** be supplied.
        docvecs_mapfile : str, optional
            Path to a file containing the docvecs mapping. If `docvecs` is None, this file will be used to create it.
        comment : str, optional
            A model descriptive comment, used for logging and debugging purposes.
        trim_rule : function ((str, int, int) -> int), optional
            Vocabulary trimming rule that accepts (word, count, min_count).
            Specifies whether certain words should remain in the vocabulary (:attr:`gensim.utils.RULE_KEEP`),
            be trimmed away (:attr:`gensim.utils.RULE_DISCARD`), or handled using the default
            (:attr:`gensim.utils.RULE_DEFAULT`).
            If None, then :func:`gensim.utils.keep_vocab_item` will be used.
        size : int, optional
            Dimensionality of the feature vectors.
        alpha : float, optional
            The initial learning rate.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`.
            Note that for a **fully deterministically-reproducible run**, you **must also limit the model to
            a single worker thread (`workers=1`)**, to eliminate ordering jitter from OS thread scheduling.
            In Python 3, reproducibility between interpreter launches also requires use of the `PYTHONHASHSEED`
            environment variable to control hash randomization.
        workers : int, optional
            Use this many worker threads to train the model. Will yield a speedup when training with multicore machines.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        hs : int {1,0}, optional
            If 1, hierarchical softmax will be used for model training. If set to 0, and `negative` is non-zero,
            negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
        cbow_mean : int, optional
            Same as `dm_mean`, **unused**.
        hashfxn : function (object -> int), optional
            A hashing function. Used to create an initial random reproducible vector by hashing the random seed.
        iter : int, optional
            Number of epochs to iterate through the corpus.
        sorted_vocab : bool, optional
            Whether the vocabulary should be sorted internally.
        batch_words : int, optional
            Number of words to be handled by each job.
        """
        self.gensim_model = None
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.comment = comment
        self.trim_rule = trim_rule

        # attributes associated with gensim.models.Word2Vec
        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.epochs = epochs
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {iterable of :class:`~gensim.models.doc2vec.TaggedDocument`, iterable of list of str}
            A collection of tagged documents used for training the model.
        Returns
        -------
        :class:`~gensim.sklearn_api.d2vmodel.D2VTransformer`
            The trained model.
        """
        X_np = np.asarray(X, dtype = 'str')
        X = np.char.split(X_np, ' ')
        
        if isinstance([i for i in X[:1]][0], doc2vec.TaggedDocument):
            d2v_sentences = X
        else:
            d2v_sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(X)]
            
        self.gensim_model = models.Doc2Vec(
            documents=d2v_sentences, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            comment=self.comment,
            trim_rule=self.trim_rule, vector_size=self.vector_size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, sample=self.sample,
            seed=self.seed, workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
            negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn,
            epochs=self.epochs, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )

        return self

    def transform(self, docs):
        """Infer the vector representations for the input documents.
        Parameters
        ----------
        docs : {iterable of list of str, list of str}
            Input document or sequence of documents.
        Returns
        -------
        numpy.ndarray of shape [`len(docs)`, `size`]
            The vector representation of the `docs`.
        """
        
        docs_np = np.asarray(docs, dtype = 'str')
        docs = np.char.split(docs_np, ' ')

        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(docs[0], string_types):
            docs = [docs]
        vectors = [self.gensim_model.infer_vector(doc) for doc in docs]
        return np.reshape(np.array(vectors), (len(docs), self.gensim_model.vector_size))