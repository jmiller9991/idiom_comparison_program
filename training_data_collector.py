from transformers import pipeline
from huggingface_hub import login
import sys
import pandas as pd
import numpy as np
import time
import requests

MODEL_NAME_GPT2 = 'gpt2'
MODEL_NAME_LLAMA3 = 'meta-llama/Meta-Llama-3-8B' # CUDA memory issues
MODEL_NAME_LLAMA2 = 'meta-llama/Llama-2-7b-hf'
MODEL_NAME_FALCON = 'tiiuae/falcon-7b' # FALCON model caused memory error
MODEL_NAME_DEEPSEEK3 = 'deepseek-ai/DeepSeek-V3' # Deepseek threw error "ValueError: FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100), actual = `6.1`"
MODEL_NAME_DEEPSEEK2 = 'deepseek-ai/DeepSeek-V2' # Deepseek threw error "ValueError: FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100), actual = `6.1`"
MODEL_NAME_BITNET = 'microsoft/bitnet-b1.58-2B-4T' # failed to collect proper files; weird config
MODEL_NAME_QWEN = 'Qwen/Qwen3-32B' # failed to download proper files; very large
MODEL_NAME_GEMMA= 'google/gemma-3-1b-it'
MODEL_NAME_HELIUM = 'kyutai/helium-1-2b' # model throws that it can't do that any time it runs

# helper to convert array to csv
def array_to_csv(array):
    print(array)

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
        question = f"Can you translate \"{i}\" into {language}? Please format this by just writing the translation and nothing else."
        #question = f"The real meaning of \"{i}\" in {language} is: "
        response = generator(question, max_length=200)

        result_array.append(response[0]['generated_text'])

        time.sleep(2)

    return result_array

# get data for one answer to test connectivity
def test_model(translation_array, model, token):
    login(token=token)
    try:
        generator = pipeline('text-generation', model=model) #, tokenizer=model)

        question = f"Can you translate the phrase \"{translation_array[0]}\" into English? Please write as the original text separated by a colon and then the translated text."
        #question = f"The real meaning of \"{translation_array[0]}\" in English is: "
        response = generator(question, max_length=1000)

        print(response[0]['generated_text'])
    except Exception as e:
        print(f'Model Failed!\n{e}')


# converts the answer response to a csv file
def convert_answers_array(answer_array, ref_array):
    new_array = []
    new_array = answer_array + ref_array[0]

    print(new_array)

# main program execution
def main():
    if len(sys.argv) > 1:
        token = token_grabber(sys.argv[1])
    else:
        token = token_grabber()

    spanish_data, english_data = load_data_for_translation('comma_delimited_data.csv')
    test_model(spanish_data, MODEL_NAME_DEEPSEEK2, token)

    # # gpt2 array
    # gpt_array = gather_data_from_ai(spanish_data, 'English', MODEL_NAME_GPT2, token)
    #
    # # Deepseek array
    # deepseek_array = gather_data_from_ai(spanish_data, 'English', MODEL_NAME_DEEPSEEK, token)
    #
    # # Gemma array
    # gemma_array = gather_data_from_ai(spanish_data, 'English', MODEL_NAME_GEMMA, token)
    #
    # # process all arrays and convert to csv

    #exp_array = gather_data_from_ai(spanish_data, 'English', MODEL_NAME_GPT2, token)




if __name__ == '__main__':
    main()

