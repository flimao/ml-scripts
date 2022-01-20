#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import re
from unidecode import unidecode

# Pipeline - Text Preprocessing

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

# obter radicais via stemmer ou lemmatizer
def get_radicals(word_list, radicalizer):
    radicalized_words = []
    for w in word_list:
        r_words = radicalizer(w)
        radicalized_words.append(r_words)
    
    return radicalized_words

# para o spacy, as funções são aplicadas em uma ordem e de forma diferentes.
# criamos uma função específica para análise com spacy
def tokenize_remove_stopwords_get_radicais_spacy(word_list, nlp, stopword_list = None, retornar_string = True):
    
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


# aplicação do pipe
def preprocessing(string, preproc_funs_args):
    
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