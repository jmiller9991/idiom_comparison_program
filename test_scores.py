import os
import sys
import nltk
from bert_score import BERTScorer, score
import pandas as pd
import numpy as np

# easy split string method
def splitstr(sentence):
    return sentence.split(' ')

# sub-parser for BERTScore
# result should be in form of ['<String input>']
def bertscore_subprocessor(string_to_mod):
    result = [string_to_mod]
    return result

# sub-parser for BLEU score
# result should be split by spaces ['<String>', '<Input>']
def bleuscore_subprocessor(string_to_mod):
    result = splitstr(string_to_mod)
    return result

# get the csv file as input and break it down to numpy array
def loadAndParse(file):
    # import data
    data_frame_corpus = pd.read_csv(file)

    # get language arrays from dataframe
    spanish_data = data_frame_corpus['Spanish'].tolist()
    english_data = data_frame_corpus['English (Metaphorical)'].tolist()

    # loop through spanish data and make BERTScore data
    spanish_data_bert = []
    for i in spanish_data:
        spanish_data_bert.append(bertscore_subprocessor(i))

    # loop through english data and make BERTScore data
    english_data_bert = []
    for i in english_data:
        english_data_bert.append(bertscore_subprocessor(i))

    # loop through spanish data and make BLEU Score data
    spanish_data_bleu = []
    for i in spanish_data:
        spanish_data_bleu.append(bleuscore_subprocessor(i))


    # loop through english data and make BLEU Score data
    english_data_bleu = []
    for i in english_data:
        english_data_bleu.append(bleuscore_subprocessor(i))

    return spanish_data_bert, english_data_bert, spanish_data_bleu, english_data_bleu

# get the BERTScore for English
def perform_BERTScore(gen, ref, lang):
    precision, recall, f1_score = score(gen, ref, lang=lang)
    return precision, recall, f1_score

# get the BLEU score for English
def perform_BLEUScore(gen, ref):
    bleuscore = nltk.translate.bleu_score.sentence_bleu([ref], gen)
    return bleuscore

# main to run the program
def main():
    # load and parse the comma delimited corpus data as a csv file
    es_ref_array_bert, en_ref_array_bert, es_ref_array_bleu, en_ref_array_bleu = loadAndParse('comma_delimited_data.csv')
    print(es_ref_array_bert)
    print(en_ref_array_bert)
    print(es_ref_array_bleu)
    print(en_ref_array_bleu)


if __name__ == '__main__':
    main()


# BLEU test
# reference = splitstr('The quick brown fox jumps over the lazy dog')
# bad_candidate = splitstr('The quack brown fox jumps over the lazy dog')
# really_bad_candidate = splitstr('Bad Example')
#
# good_bleu = perform_BLEUScore(reference, reference)
# bad_bleu = perform_BLEUScore(bad_candidate, reference)
# really_bad_bleu = perform_BLEUScore(really_bad_candidate, reference)
#
# print(good_bleu)
# print(bad_bleu)
# print(really_bad_bleu)

# BERTScore test
# really_bad_candidate = ['BAD EXAMPLE']
# bad_canidate = ['The quack brown fox jumps over the lazy dog']
# reference = ['The quick brown fox jumps over the lazy dog']
#
# good_p, good_r, good_f1 = perform_BERTScore(reference, reference, 'en')
# bad_p, bad_r, bad_f1 = perform_BERTScore(bad_canidate, reference, 'en')
# rbad_p, rbad_r, rbad_f1 = perform_BERTScore(really_bad_candidate, reference, 'en')
#
# print(f'Good test: P: {good_p.mean().item()}, R: {good_r.mean().item()}, F1: {good_f1.mean().item()}')
# print(f'Bad test: P: {bad_p.mean().item()}, R: {bad_r.mean().item()}, F1: {bad_f1.mean().item()}')
# print(f'Really Bad test: P: {rbad_p.mean().item()}, R: {rbad_r.mean().item()}, F1: {rbad_f1.mean().item()}')
