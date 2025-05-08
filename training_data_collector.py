from transformers import pipeline
from huggingface_hub import login
import sys
import pandas as pd
import numpy as np
import time

MODEL_NAME_GPT2 = 'gpt2'
MODEL_NAME_LLAMA = 'meta-llama/Meta-Llama-3-8B'
MODEL_NAME_FALCON = 'tiiuae/falcon-7b'

# read token from token_hugging_face.txt
def token_grabber(possible_token=None):
    token = ""

    if possible_token is not None:
        token = possible_token
    else:
        with open('token_hugging_face.txt', 'r') as f:
            token = f.readline()

    return token


# loads the data as lists to translate and loop
def load_data_for_translation(file):
    data_frame_corpus = pd.read_csv(file)

    # get language arrays from dataframe
    spanish_data = data_frame_corpus['Spanish'].tolist()
    english_data = data_frame_corpus['English (Metaphorical)'].tolist()

    return spanish_data, english_data

# this method gathers data from an AI, translating all values from corpus to language provided by user
def gather_data_from_ai(translation_array, language, model, token):
    result_array = []

    login(token=token)

    generator = pipeline('text-generation', model=model)

    for i in translation_array:
        question = f"Can you translate {i} into {language}?"
        response = generator(question, max_new_tokens=50)

        print(response[0]['generated_text'])

        result_array.append(response[0]['generated_text'])

        time.sleep(2)

    return result_array

# get data for one answer to test connectivity
def get_answer_from_ai(translation_array, token):
    login(token=token)

    generator = pipeline('text-generation', model=MODEL_NAME_GPT2)

    question = f"Can you translate {translation_array[0]} into English?"
    response = generator(question, max_new_tokens=50)

    print(response[0]['generated_text'])


# converts the answer response to a csv file
def convert_answers_array(answer_array, model):
    if model == MODEL_NAME_GPT2:
        print(answer_array)
    elif model == MODEL_NAME_LLAMA:
        print(answer_array)
    elif model == MODEL_NAME_FALCON:
        print(answer_array)

# main program execution
def main():
    if len(sys.argv) > 1:
        token = token_grabber(sys.argv[1])
    else:
        token = token_grabber()

    spanish_data, english_data = load_data_for_translation('comma_delimited_data.csv')
    get_answer_from_ai(spanish_data, token)


if __name__ == '__main__':
    main()

