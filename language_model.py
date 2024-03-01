import pandas as pd
import numpy as np
import re
import argparse
import string
from collections import defaultdict, Counter
from scipy.stats import linregress
import math
from tqdm import tqdm

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


def flatten_corpus(corpus): # This function will take corpus which consist of list of list and will return a list of words
    flattened_corpus = []
    for sentence in corpus:
        for word in sentence:
            flattened_corpus.append(word)


    return flattened_corpus


class N_Gram_model:
    '''
    self.list_n_grams: list of all n-grams

    '''

    def __init__(self, corpus_path, lm_type = 'g', N=3):
        self.lm_type = lm_type
        self.path = corpus_path
        self.N = N

    # Read the file and store the content in a variable
    def read_file(self):
        with open(self.path, 'r', encoding='utf-8') as file:
            self.corpus_content = file.read()

    # Tokenize the content of the file and remove all the punctuation marks
    def preprocess(self):
        self.tokenized_sentences = tokenize(self.corpus_content)
        self.tokenized_sentences = [[word for word in inner_list if word not in string.punctuation] for inner_list in self.tokenized_sentences]

    # Divide the corpus into training and test sets
    def set_up(self):
        temp = self.tokenized_sentences
        np.random.shuffle(temp)
        size = 1000 #
        self.test = temp[:size]
        self.val = temp[size:2*size]
        self.train = temp[2*size:]

    def save(self, prob_dict):
        # Use np.save to save the numpy array to a file
        np.save('Ulysses_prob_dict_good_turing_new.npy', prob_dict)
        

    def load(self, path):
        # Use np.load to load the numpy array from a file
        loaded_dict = np.load(path, allow_pickle=True).item()
        self.probab_dict = loaded_dict

    def get_tuple_list(self,sentence, N):
        num_tokens = len(sentence)
        for i in range(N-1):
            sentence.insert(0, '<s>')

        sentence.append('</s>')
        n_gram = defaultdict(list)
        i = 0
        for word in sentence:
            if word == '<s>':
                i += 1
                continue
            else:
                # key will be the preceeding n words and value will be the word
                context = tuple(sentence[i-N+1:i])
                target_word = word
                i += 1

            # append the key value pair to n_gram dictionary
            n_gram[context].append(target_word)

        grams = []
        for key, value in n_gram.items():
            for v in value:
                grams.append(list(key) + [v])

        # Creating a frequency dictionary
        tuple_list = [tuple(inner_list) for inner_list in grams]
        return tuple_list

    def n_grams(self, corpus, N):
        flattened_corpus = flatten_corpus(corpus)
        for i in range(N-1):
            flattened_corpus.insert(0, '<s>')

        flattened_corpus.append('</s>')
    
        n_gram = defaultdict(list)
        i = 0
        for word in flattened_corpus:
            if word == '<s>':
                i += 1
                continue
            else:
                # key will be the preceeding n words and value will be the word
                context = tuple(flattened_corpus[i-N+1:i])
                target_word = word
                i += 1

            # append the key value pair to n_gram dictionary
            n_gram[context].append(target_word)
        
        return n_gram

    def get_slope_intercept(self, freq_of_freq_dict):
        # Sort the dictionary
        sorted_freq_of_freq_dict = sorted(freq_of_freq_dict.keys())

        # Fill the missing values with 0
        new_freq_of_freq_dict = {key: freq_of_freq_dict.get(key, 0) for key in range(sorted_freq_of_freq_dict[0],sorted_freq_of_freq_dict[-1] + 1)}
        # print(new_freq_of_freq_dict)

        # Get the non zero keys
        non_zero_keys = [key for key, value in new_freq_of_freq_dict.items() if value > 0]

        # Calculate the Zrs
        Zrs = {}
        for num in non_zero_keys:
            if num - 1 > 0 and num + 1 < len(non_zero_keys):
                # index of num
                index = non_zero_keys.index(num)
                q = non_zero_keys[index - 1]
                t = non_zero_keys[index + 1]
                Zr = new_freq_of_freq_dict[num]*2 / (t - q)
                Zrs[num] = Zr
            else:
                Zrs[num] = new_freq_of_freq_dict[num]
            
        # Calculate the log of Zrs
        log_r = np.log10(list(Zrs.keys()))
        log_Zr = np.log10(list(Zrs.values()))       
        slope, intercept, r_value, p_value, std_err = linregress(log_r, log_Zr)
        return slope, intercept, new_freq_of_freq_dict


    def good_turing(self):
        # self.n_gram = self.n_grams(self.N)
        grams = []
        for key, value in self.n_gram.items():
            for v in value:
                grams.append(list(key) + [v])

        # Creating a frequency dictionary
        tuple_list = [tuple(inner_list) for inner_list in grams]
        self.list_n_grams = tuple_list
        # print("tuple list: ", tuple_list)
        list_counts = Counter(tuple_list)
        list_counts_dict = dict(list_counts)
            
        # Creating a frequency of frequency dictionary
        freq_of_freq = Counter(list_counts_dict.values())
        freq_of_freq_dict = dict(freq_of_freq)
        # print("count list: ")
        # for key, value in list_counts_dict.items():
        #     print(key, value)

        # print("freq of freq: ")
        # for key, value in freq_of_freq_dict.items():
        #     print(key, value)

        N = 0 
        for key, value in freq_of_freq_dict.items():
            N += (key*value)
            # print("key: ", key)
            # print("value: ", value)
            # print("N: ", N)
            

        slope, intercept, new_freq_of_freq_dict = self.get_slope_intercept(freq_of_freq_dict)
        r_stars = {}
        for num in new_freq_of_freq_dict.keys():
            r_star = ((num + 1)**(slope +1))/(num**(slope))
            r_stars[num] = r_star


        # print("r_stars: ")
        # for key, value in r_stars.items():
        #     print(key, value)
        # print("length of r_stars: ", len(r_stars))
        # print("n gram model:", self.n_gram)
        # Calculate the probability of each n-gram
        context = defaultdict(list)
        for key, value in tqdm(list_counts_dict.items(), desc="Processing context", unit="n-gram"):
            c = list_counts_dict[key]
            context[tuple(list(key)[:2])].append(r_stars[c])

        print("context: ", context)
        for key, value in context.items():
            context[key] = sum(value)

        

        prob_dict = {}
        for key, value in tqdm(list_counts_dict.items(), desc="Processing n-gram probab", unit="n-gram"):
            r_star = r_stars[value]  # New count for the n-gram
            # print("key: ", key)
            # print("tuple", tuple(list(key)[:2]))
            # print("list: ",self.n_gram[tuple(list(key)[:2])])
            # count =  len(self.n_gram[tuple(list(key)[:2])]) # Count of occurrence of context
            partial_key = tuple(list(key)[:2])
            count = 0
            if partial_key in context.keys():
                count = context[partial_key]
            else:
                print("Not in context")
            # print("r_star: ", r_star)
            # print("count: ", count)
            # print("--"*10)
            # print("partial key: ", partial_key)
            # print("count: ", count)
            # print("r_star: ", r_star)
            # print("probab: ", r_star /  count)
            prob_dict[key] = r_star /  count
            
        # Probability of unseen n-grams
        unseen_probab = freq_of_freq_dict[1]/N
        # print("N: ", N)
        # print("freq of freq: ", freq_of_freq_dict[1])
        # print("unseen probab: ", unseen_probab)
        # Add this probability to the prob_dict
        prob_dict[('<UNK>',)] = unseen_probab
        # print("prob dict: ")
        # for key, value in prob_dict.items():
        #     print(key," : " ,value)

        self.save(prob_dict)
        
    def get_count_dict(self, model):
        grams = []
        for key, value in model.items():
            for v in value:
                grams.append(list(key) + [v])

        # Creating a frequency dictionary
        tuple_list = [tuple(inner_list) for inner_list in grams]
        list_counts = Counter(tuple_list)
        list_counts_dict = dict(list_counts)

        return list_counts_dict
    

    def get_lambda(self):
        # print("in get lambda function")
        model_tri = self.n_grams(self.val, 3)
        model_bi = self.n_grams(self.val, 2)
        model_uni = self.n_grams(self.val, 1)

        count_list_tri = self.get_count_dict(model_tri)
        count_list_bi = self.get_count_dict(model_bi)
        count_list_uni = self.get_count_dict(model_uni)

        l = [item for sublist in list(model_uni.values()) for item in sublist]
        # print("l: ", l)
        num_tokens = len(l)

        l1, l2, l3 = 0, 0, 0

        for inner_list, count in count_list_tri.items():
            list_tup = list(inner_list)
            if tuple(list_tup[0:2]) == ('<s>', '<s>'):
                continue
            if(count > 0):
                list_tup = list(inner_list)
                f123 = count
                # print("count list bi: ", count_list_bi)
                f12 = count_list_bi[tuple(list_tup[0:2])]
                f23 = count_list_bi[tuple(list_tup[-2:])]
                f2 = l.count(list_tup[-2])
                f3 = l.count(list_tup[-1])

                if f12 > 1:    
                    case1 = (f123 - 1)/(f12 - 1)
                else:
                    case1 = 0

                if f2 > 1:
                    case2 = (f23 - 1)/(f2 - 1)
                else:
                    case2 = 0

                if num_tokens > 1:
                    case3 = (f3 - 1)/(num_tokens - 1)
                else:
                    case3 = 0
                
                # If denominator is 0, then the value of case is also 0
                if case1 == max(case1, case2, case3):
                    l3 += count
                elif case2 == max(case1, case2, case3):
                    l2 += count
                else:
                    l1 += count
            
        # Normalise the lambdas
        sum = l1 + l2 + l3    
        l1 = l1 / sum
        l2 = l2 / sum
        l3 = l3 / sum
        # print("l1: ", l1)
        # print("l2: ", l2)
        # print("l3: ", l3)
        return l1, l2, l3

        
    def interpolation(self):
        # Get the lambdas
        l1, l2, l3 = self.get_lambda()
        model_tri = self.n_grams(self.train, 3)
        model_bi = self.n_grams(self.train, 2)
        model_uni = self.n_grams(self.train, 1)
        model_uni = list(model_uni.values())[0]
        
        grams = []
        for key, value in self.n_gram.items():
            for v in value:
                grams.append(list(key) + [v])

        # Creating a frequency dictionary
        tuple_list = [tuple(inner_list) for inner_list in grams]
        self.list_n_grams = tuple_list
        # print("tuple list: ", tuple_list)
        list_counts = Counter(tuple_list)
        # This list contains all the trigram sequences. We will use its keys
        list_counts_dict = dict(list_counts) 

        interpolation_probab_dict = {}
        for key in list_counts_dict.keys():
            print("key: ", key)
            context_tri = tuple(list(key)[:2])
            context_bi = tuple(list(key)[1:2])
            context_uni = tuple(list(key)[-1:])
            target_word = list(key)[-1:][0]
            p_tri = l1 * (model_tri[context_tri].count(target_word)/len(model_tri[context_tri]))
            p_bi = l2 * (model_bi[context_bi].count(target_word)/len(model_bi[context_bi]))
            p_uni = l3 * (model_uni.count(target_word)/len(model_uni))

            total_probab = p_tri + p_bi + p_uni
            interpolation_probab_dict[key] = total_probab
        
        self.save(interpolation_probab_dict)

    def training(self):
        self.n_gram = self.n_grams(self.train, self.N)
        
        if self.lm_type == 'g':
            self.good_turing()
        elif self.lm_type == 'i':
            self.interpolation()
        

    def perplexity(self, sentence, l1, l2, l3, model_tri, model_bi, model_uni):
        # print("sentence: ", sentence)
        num_tokens = len(sentence)
        # print("num tokens: ", num_tokens)
        for i in range(self.N-1):
            sentence.insert(0, '<s>')

        sentence.append('</s>')
        n_gram = defaultdict(list)
        i = 0
        for word in sentence:
            if word == '<s>':
                i += 1
                continue
            else:
                # key will be the preceeding n words and value will be the word
                context = tuple(sentence[i-self.N+1:i])
                target_word = word
                i += 1

            # append the key value pair to n_gram dictionary
            n_gram[context].append(target_word)

        grams = []
        for key, value in n_gram.items():
            for v in value:
                grams.append(list(key) + [v])

        # Creating a frequency dictionary
        tuple_list = [tuple(inner_list) for inner_list in grams]
        # print("tuple list: ", tuple_list)
        perplexity = 1
        if self.lm_type == 'g':
            for n_gram in tuple_list:
                if n_gram in self.probab_dict.keys():
                    perplexity *= (1/self.probab_dict[n_gram])
                else:
                    perplexity *= (1/self.probab_dict[('<UNK>',)])
        elif self.lm_type == 'i':
            # l1, l2, l3 = self.get_lambda()
            # model_tri = self.n_grams(self.train, 3)
            # model_bi = self.n_grams(self.train, 2)
            # model_uni = self.n_grams(self.train, 1)
            # model_uni = list(model_uni.values())[0]
            
            for n_gram in tuple_list:
                if n_gram in self.probab_dict.keys():
                    perplexity *= (1/self.probab_dict[n_gram])
                else:
                    context_tri = tuple(list(n_gram)[:2])
                    context_bi = tuple(list(n_gram)[1:2])
                    context_uni = tuple(list(n_gram)[-1:])
                    target_word = list(n_gram)[-1:][0]

                    if len(model_tri[context_tri]) == 0:
                        p_tri = 0
                    else:
                        p_tri = l1 * (model_tri[context_tri].count(target_word)/len(model_tri[context_tri]))
                    
                    if len(model_bi[context_bi]) == 0:
                        p_bi = 0
                    else:
                        p_bi = l2 * (model_bi[context_bi].count(target_word)/len(model_bi[context_bi]))
                        
                    p_uni = l3 * (model_uni.count(target_word)/len(model_uni))

                    total_probab = p_tri + p_bi + p_uni

                    if total_probab == 0:
                        total_probab = 1e-5
                    
                    perplexity *= (1/total_probab)

        
        perplexity = np.power(perplexity, 1/num_tokens)
        return perplexity

    def write_perplexity(self):
        perplexity_train = {}
        # print("num sentences in train: ", len(self.train))
        # i = 0
        model_tri = self.n_grams(self.train, 3)
        model_bi = self.n_grams(self.train, 2)
        model_uni = self.n_grams(self.train, 1)
        model_uni = list(model_uni.values())[0]
        l1, l2, l3 = self.get_lambda()
        for sentence in tqdm(self.train, desc="Processing n-grams train", unit="n-gram"):
            sent_joined = " ".join(sentence)
            if sentence == []:
                continue

            perplexity = self.perplexity(sentence, l1, l2, l3, model_tri, model_bi, model_uni)

            perplexity_train[sent_joined] = perplexity
    

        perplexity_train_avg = sum(perplexity_train.values())/len(perplexity_train)
        print("perplexity_train_avg: ", perplexity_train_avg)

        file_path = '2022121003_LM2_train-perplexity.txt'

        # Write the value to the file
        with open(file_path, 'w') as file:
            file.write(f"Average perplexity: {perplexity_train_avg}\n")

        # Append key-value pairs to the file
        with open(file_path, 'a') as file:
            for key, value in perplexity_train.items():
                file.write(f"{key}:\t {value}\n")

        perplexity_test = {}
        for sentence in tqdm(self.test, desc="Processing n-grams test", unit="n-gram"):
            sent_joined = " ".join(sentence)
            if sentence == []:
                continue
            perplexity = self.perplexity(sentence, l1, l2, l3, model_tri, model_bi, model_uni)

            perplexity_test[sent_joined] = perplexity

        perplexity_test_avg = sum(perplexity_test.values())/len(perplexity_test)
        print("perplexity_test_avg: ", perplexity_test_avg)

        file_path = '2022121003_LM2_test-perplexity.txt'

        # Write the value to the file
        with open(file_path, 'w') as file:
            file.write(f"Average perplexity: {perplexity_test_avg}\n")

        # Append key-value pairs to the file
        with open(file_path, 'a') as file:
            for key, value in perplexity_test.items():
                file.write(f"{key}:\t {value}\n")

    def evaluate(self, sentence):
        tokenize_sentence = tokenize(sentence)
        flattened_corpus = flatten_corpus(tokenize_sentence)
        for i in range(self.N-1):
            flattened_corpus.insert(0, '<s>')

        flattened_corpus.append('</s>')
    
        n_gram = defaultdict(list)
        i = 0
        for word in flattened_corpus:
            if word == '<s>':
                i += 1
                continue
            else:
                # key will be the preceeding n words and value will be the word
                context = tuple(flattened_corpus[i-self.N+1:i])
                target_word = word
                i += 1

            # append the key value pair to n_gram dictionary
            n_gram[context].append(target_word)

        grams = []
        for key, value in n_gram.items():
            for v in value:
                grams.append(list(key) + [v])

        # Creating a frequency dictionary
        tuple_list = [tuple(inner_list) for inner_list in grams]
        sentence_n_gram = tuple_list
        # print("sentence n gram: ", sentence_n_gram)
        
        probab_score = 0

        if self.lm_type == 'g':
            for n_gram in sentence_n_gram:
                # print("n gram: ", n_gram)
                if n_gram in self.probab_dict.keys():
                    probab_score += np.log10(self.probab_dict[n_gram])
                    # print("probab score of ngram: ", self.probab_dict[n_gram])
                else:
                    probab_score += np.log10(self.probab_dict[('<UNK>',)])
                    # print("probab score of ngram: ", self.probab_dict[('<UNK>',)])
        elif self.lm_type == 'i':
            l1, l2, l3 = self.get_lambda()
            model_tri = self.n_grams(self.train, 3)
            model_bi = self.n_grams(self.train, 2)
            model_uni = self.n_grams(self.train, 1)
            model_uni = list(model_uni.values())[0]
            
            for n_gram in sentence_n_gram:
                # print("n gram: ", n_gram)
                if n_gram in self.probab_dict.keys():
                    # print("total probab of ngram: ", self.probab_dict[n_gram])
                    probab_score += np.log10(self.probab_dict[n_gram])
                    # probab_score *= (self.probab_dict[n_gram])

                else:
                    context_tri = tuple(list(n_gram)[:2])
                    context_bi = tuple(list(n_gram)[1:2])
                    context_uni = tuple(list(n_gram)[-1:])
                    target_word = list(n_gram)[-1:][0]

                    if len(model_tri[context_tri]) == 0:
                        p_tri = 0
                    else:
                        p_tri = l1 * (model_tri[context_tri].count(target_word)/len(model_tri[context_tri]))
                    
                    if len(model_bi[context_bi]) == 0:
                        p_bi = 0
                    else:
                        p_bi = l2 * (model_bi[context_bi].count(target_word)/len(model_bi[context_bi]))
                        
                    p_uni = l3 * (model_uni.count(target_word)/len(model_uni))

                    total_probab = p_tri + p_bi + p_uni

                    if total_probab == 0:
                        total_probab = 1e-5
                    
                    # print("total probab of ngram: ", total_probab)
                    probab_score += np.log10(total_probab)
                    # probab_score *= (total_probab)
                # print("-"*10)

        probab_score = np.power(10, probab_score)
        print("probab score: ", probab_score)

    def generate(self, model, context, N=3):
        tokenized_context = tokenize(context)
        n = N - 1


        tokenized_context = tuple(tokenized_context[-1][-n:])

        if tokenized_context not in model.keys():
            return '</s>'
        else:
            next_word = max(set(model[tokenized_context]), key=model[tokenized_context].count)
            return next_word

    def generate_sequence(self, model, context, N=3):
        if N <= 0:
            return "N should be greater than 0"
        
        generated_sequence = []
        generated_sequence.append(context)
        max_tokens = 20
        for i in range(max_tokens):
            # print("context", context)
            next_word = self.generate(model, context, N)
            generated_sequence.append(next_word)
            context = context.split()
            context = context[1:]
            
            context.append(next_word)
            context = ' '.join(context)
            if next_word == '</s>':
                break
            

        return " ".join(generated_sequence)
    
    def next_k_words(self, context, k):
        tokenized_context = tokenize(context)
        n = self.N - 1

        tokenized_context = tokenized_context[-1][-n:]
        filtered_trigrams = {trigram: prob for trigram, prob in self.probab_dict.items() if trigram[:2] == tuple(tokenized_context)}
        
        sorted_trigrams = dict(sorted(filtered_trigrams.items(), key=lambda x: x[1], reverse=True))
        top_k_trigrams = dict(list(sorted_trigrams.items())[:k])

        if top_k_trigrams != {}:
            for key, value in top_k_trigrams.items():
                word = list(key)[-1]
                print(word,'\t', value)
        else:
            if self.lm_type == 'g':
                print('<UNK>','\t',self.probab_dict[('<UNK>',)])
            elif self.lm_type == 'i':
                print('<UNK>','\t','0.0001')

    def generate_smooth(self, context, N=3):
        tokenized_context = tokenize(context)
        n = N - 1


        tokenized_context = tokenized_context[-1][-n:]
        filtered_trigrams = {trigram: prob for trigram, prob in self.probab_dict.items() if trigram[:2] == tuple(tokenized_context)}
        
        sorted_trigrams = dict(sorted(filtered_trigrams.items(), key=lambda x: x[1], reverse=True))
        # print("sorted trigrams: ", sorted_trigrams)
        if sorted_trigrams == {}:
            return '<UNK>'
        else:
            top_trigrams = list(sorted_trigrams.keys())[0]
            next_word = top_trigrams[-1]
        # print("top trigram: ", top_trigrams)
        # print("next word: ", next_word)

        return next_word

    def generate_smooth_sequence(self, context, N=3):
        if N <= 0:
            return "N should be greater than 0"
        
        generated_sequence = []
        generated_sequence.append(context)
        max_tokens = 20
        for i in range(max_tokens):
            # print("context", context)
            next_word = self.generate_smooth(context, N)
            generated_sequence.append(next_word)
            context = context.split()
            context = context[1:]
            
            context.append(next_word)
            context = ' '.join(context)
            if next_word == '</s>':
                break
            

        return " ".join(generated_sequence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lm_type", help="type of language model to be used")
    parser.add_argument("corpus_path", help="path to the corpus")
    args = parser.parse_args()
    corpus_path = args.corpus_path
    lm_type = args.lm_type
    # print("corpus path: ", corpus_path)
    # print("lm type: ", lm_type)

    input_sent = input("Input sentence: ")

    if lm_type == 'g' and ('Pride' in corpus_path):
        N_Gram_model = N_Gram_model(corpus_path, 'g')
        N_Gram_model.read_file()
        N_Gram_model.preprocess()
        N_Gram_model.set_up()
        N_Gram_model.load('Jane_Austen_prob_dict_good_turing_new.npy')
        N_Gram_model.evaluate(input_sent)
        
    elif lm_type == 'i' and ('Pride' in corpus_path):
        N_Gram_model = N_Gram_model(corpus_path, 'i')
        N_Gram_model.read_file()
        N_Gram_model.preprocess()
        N_Gram_model.set_up()
        N_Gram_model.load('Jane_Austen_prob_dict_interpolation.npy')
        N_Gram_model.evaluate(input_sent)
        
    elif lm_type == 'g' and ('Ulysses' in corpus_path):
        N_Gram_model = N_Gram_model(corpus_path, 'g')
        N_Gram_model.read_file()
        N_Gram_model.preprocess()
        N_Gram_model.set_up()
        N_Gram_model.load('Ulysses_prob_dict_good_turing_new.npy')
        N_Gram_model.evaluate(input_sent)
        
    elif lm_type == 'i' and ('Ulysses' in corpus_path):
        N_Gram_model = N_Gram_model(corpus_path, 'i')
        N_Gram_model.read_file()
        N_Gram_model.preprocess()
        N_Gram_model.set_up()
        N_Gram_model.load('Ulysses_prob_dict_interpolation.npy')
        N_Gram_model.evaluate(input_sent)
        
