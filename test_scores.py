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

# this loads and parses the simplified answer lists
def load_and_parse_simplified(file):
    # import data
    data = pd.read_csv(file)

    # get language data
    spanish = data['Spanish'].tolist()
    english = data['English Translation'].tolist()

    # loop through spanish and english, processing for BERTScore
    es_bert = []
    for i in spanish:
        es_bert.append(bertscore_subprocessor(i))

    en_bert = []
    for i in english:
        en_bert.append(bertscore_subprocessor(i))

    # loop through spanish and english, processing for BLEU score
    es_bleu = []
    for i in spanish:
        es_bleu.append(bleuscore_subprocessor(i))

    en_bleu = []
    for i in english:
        en_bleu.append(bleuscore_subprocessor(i))

    return es_bert, en_bert, es_bleu, en_bleu

# get the BERTScore for English
def perform_BERTScore(gen, ref, lang):
    precision, recall, f1_score = score(gen, ref, lang=lang)
    return precision, recall, f1_score

# get the BLEU score for English
def perform_BLEUScore(gen, ref):
    bleuscore = nltk.translate.bleu_score.sentence_bleu([ref], gen)
    return bleuscore

# loop through lists  and give BLEU score
def bleuscore_loop(cand_list, ref_list):
    results = []
    for i in range(0, len(cand_list)-1):
        results.append(perform_BLEUScore(cand_list[i], ref_list[i]))

    return results

# loop through list and give BERTScore
def bertscore_loop(cand_list, ref_list, lang):
    results = []
    for i in range(0, len(cand_list)-1):
        precision, recall, f1 = perform_BERTScore(cand_list[i], ref_list[i], lang)
        results.append([precision.item(), recall.item(), f1.item()])

    return results

# converts bert values to dataframe and writes it as a csv
def convert_and_write_bert(array, name):
    # convert to dataframe
    df = pd.DataFrame(array, columns=['precision', 'recall', 'f1'])

    df.to_csv(name+'.csv', index=False)

# converts bleu values to dataframe and writes it as a csv
def convert_and_write_bleu(array, name):
    df = pd.DataFrame(array, columns=['BLEU Score'])

    df.to_csv(name+'.csv', index=False)

# main to run the program
def main():
    # load and parse the comma delimited corpra data as a csv file
    _, en_bert_ref, _, en_bleu_ref = load_and_parse_simplified('condensed-ref.csv')
    _, en_bert_gpt, _, en_bleu_gpt = load_and_parse_simplified('gpt4-data.csv')
    _, en_bert_deepl, _, en_bleu_deepl = load_and_parse_simplified('deepl-data.csv')
    _, en_bert_deepseek, _, en_bleu_deepseek = load_and_parse_simplified('deepseek-data.csv')
    _, en_bert_gemini, _, en_bleu_gemini = load_and_parse_simplified('gemini-data.csv')

    # loop through all BERTScore Candidates and References and calculate
    # gpt
    gpt_bert = bertscore_loop(en_bert_gpt, en_bert_ref, 'en')
    gpt_bleu = bleuscore_loop(en_bleu_gpt, en_bleu_ref)

    # deepl
    deepl_bert = bertscore_loop(en_bert_deepl, en_bert_ref,'en')
    deepl_bleu = bleuscore_loop(en_bleu_deepl, en_bleu_ref)

    # deepseek
    deepseek_bert = bertscore_loop(en_bert_deepseek, en_bert_ref, 'en')
    deepseek_bleu = bleuscore_loop(en_bleu_deepseek, en_bleu_ref)

    # gemini
    gemini_bert = bertscore_loop(en_bert_gemini, en_bert_ref, 'en')
    gemini_bleu = bleuscore_loop(en_bleu_deepseek, en_bleu_ref)

    # convert tensors to dataframes and write to csvs
    # gpt
    convert_and_write_bert(gpt_bert, 'gpt-bert-res')
    convert_and_write_bleu(gpt_bleu, 'gpt-bleu-res')

    #deepl
    convert_and_write_bert(deepl_bert, 'deepl-bert-res')
    convert_and_write_bleu(deepl_bleu, 'deepl-bleu-res')

    #deepseek
    convert_and_write_bert(deepseek_bert, 'deepseek-bert-res')
    convert_and_write_bleu(deepseek_bleu, 'deepseek-bleu-res')

    # gemini
    convert_and_write_bert(gemini_bert, 'gemini-bert-res')
    convert_and_write_bleu(gemini_bleu, 'gemini-bleu-res')


    # es_ref_array_bert, en_ref_array_bert, es_ref_array_bleu, en_ref_array_bleu = loadAndParse('comma_delimited_data.csv')
    # print(es_ref_array_bert)
    # print(en_ref_array_bert)
    # print(es_ref_array_bleu)
    # print(en_ref_array_bleu)


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
