import pandas as pd
import numpy as np
import re
import argparse

def make_sentences(sentences):
    sents = []
    i = 0
    for sent in sentences:
        if len(sent) == 0:
            continue
        elif(sent == '"'):
            sents[i-1] = sents[i-1] + sent  
        else:
            sents.append(sent)
            i += 1

    return sents

def replace(sentence):
    sentence = re.sub(r'#\w+', '<HASHTAG>', sentence)
    sentence = re.sub(r'http\S+(?<!\.|\,|\?|\!)', '<URL>', sentence)
    sentence = re.sub(r'www.+(?<!\.|\,|\?|\!)', '<URL>', sentence)
    sentence = re.sub(r'\d+(\.\d+)?\s*%', '<PERCENTAGE>', sentence)
    sentence = re.sub(r'\b\d{2}:\d{2}\b', '<TIME>', sentence)
    sentence = re.sub(r'\b(\d{2}/\d{2}/\d{2})\b', '<DATE>', sentence)
    sentence = re.sub(r'\b(\d{2}/\d{2}/\d{4})\b', '<DATE>', sentence)
    sentence = re.sub(r'\b[\w.]+@[\w.]+\b', '<MAILID>', sentence)
    sentence = re.sub(r'\b\d+(\.*\d+)?\b', '<NUM>', sentence)
    sentence = re.sub(r'@[\w]+', '<MENTION>', sentence)
    return sentence

def split_words(arr):
    # Define a regex pattern to identify punctuation marks at the start or end
    # pattern = re.compile(r'(?:(^[.,!?"]*|\b[.,!?"]*$)|([.,!?"]{2,})|(["\']\b.*?\b["\']))')
    # pattern = re.compile(r'^(\W+)|(\W+)$') # WORKS BETTER 
    pattern = re.compile(r'^([^A-Za-z0-9<>]+)|([^A-Za-z0-9<>]+)$')
    
    result = []
    for word in arr:
        parts = re.split(pattern, word)
        result.extend(part for part in parts if part)

    return result

def split_words_with_punctuation(arr):
    # Define a regex pattern to identify two or more consecutive punctuation marks
    pattern  = re.compile(r'([.,!?"]{2,})')
    
    # Iterate through the array and split words with punctuation marks
    result = []
    for word in arr:
        parts = re.findall(pattern, word)
        
        if parts == []:
            result.append(word)
            continue
        split_parts = []
        for part in parts:
            if len(part) > 1 and all(char in '.,!?\"\' ' for char in part):
                split_parts.extend(char for char in part)
            else:
                split_parts.append(part.strip('"\''))
        result.extend(split_parts)

    return result

def remove_hy(tokenized_sentences):
    pattern = r'^[-_]+|[-_]+$'
    cleaned_lists = [[re.sub(pattern, '', word) for word in sublist]for sublist in tokenized_sentences]
    filtered_lists = [list(filter(lambda x: x != '', sublist))for sublist in cleaned_lists]
    return filtered_lists

def tokenize(text):
    # Fist tokenize it into sentences
    pat = re.compile(r'(?<=\?|\!|\.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)\s*("?)\s*("?)(?:\s+|$)') 

    matches = pat.split(text.replace('\n', ' '))

    sentences = make_sentences(matches)
    replaced_sentences = []
    for sentence in sentences:
        replaced_sentence = replace(sentence)
        replaced_sentences.append(replaced_sentence)

    # Split sentence according to white space
    words = []
    for sent in replaced_sentences:
        sent = sent.split()
        words.append(sent)

    # Split the punctuation marks from the words
    tokenized_sentences = []
    for sentence in words:
        tokenized_sentences.append(split_words_with_punctuation(split_words(sentence)))

    tokenized_sentences = remove_hy(tokenized_sentences)
    
    return tokenized_sentences


text  = input("Your text: ")
print(tokenize(text))