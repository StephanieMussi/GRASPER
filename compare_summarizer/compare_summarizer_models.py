import os
import re
import sys
import math
import torch
import evaluate
import tkinter as tk
from llama_cpp import Llama
from tkinter import filedialog
from timeit import default_timer as timer
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Get pdf file input in pop-up file manager window.
If no file is selected, terminate the program.
"""
def ask_for_file():
    print("Please upload summary in txt format:")
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(filetypes=[("TXT files", "*.txt")])
    if file_path:
        print(f"File selected: {file_path}")
    else:
        sys.exit("No file selected. Exiting...")
    return file_path


"""
Clean the extracted text: 
Remove before abstract, remove after references, remove reference number, remove tables
"""
def text_filter(text):
    abstract_index = text.find("Abstract")
    long_text = text[abstract_index:]
    references_index = long_text.find("References")
    long_text = long_text[:references_index]
    long_text = re.sub(r'\[\d+\]', '', long_text)
    lines = long_text.split('\n')
    in_table = False
    non_table_lines = []
    for line in lines:
        if line.startswith('Table'):
            in_table = True
        elif in_table:
            if len(line) > 20:
                in_table = False
                non_table_lines.append(line)
        else:
            non_table_lines.append(line)
    long_text = '\n'.join(non_table_lines)

    return long_text



"""
Get paper title from pdf file path.
It is assumed that name of the pdf file is the paper title.
"""
def get_paper_title(pdf_path):
    filename_with_extension = os.path.basename(pdf_path)
    paper_title, extension = os.path.splitext(filename_with_extension)
    return paper_title


"""
Split text into chunks with specified number of tokens.
"""
def chunk_text_into_tokens(text, tokenizer, max_tokens=1024):
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    count = len(token_chunks)
    return token_chunks, count


"""
With specified maximum summary length, calculate the min and max length for each chunk summary.
"""
def get_max_min_sum_length(num_chunk, max_text_length=3996):
    max_sum_length = math.floor(max_text_length / num_chunk)
    min_sum_length = math.floor(0.5 * max_sum_length)
    return max_sum_length, min_sum_length


"""
Join chunk summaries list into one single string
"""
def concatenate_summaries(summaries):
    return " ".join(summaries)


"""
Summarize text with BART model
"""
def summarize_chunks_bart(token_chunks, max_sum_length, min_sum_length, tokenizer, model):
    summaries = []
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        summary_ids = model.generate(**inputs, max_length=max_sum_length, min_length=min_sum_length, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def summarize_text_bart(long_text, max_text_length=3996):
    start = timer()
    tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_bart)
    max_sum_length, min_sum_length = get_max_min_sum_length(num_chunk)
    chunk_summaries = summarize_chunks_bart(text_chunks, max_sum_length, min_sum_length, tokenizer_bart, model_bart)
    final_summary = concatenate_summaries(chunk_summaries)
    end = timer()
    time = "Time used: " + str(end - start) + "\n\n"
    tokens_bart = tokenizer_bart.tokenize(final_summary)
    number_of_tokens = "Number of tokens: " + str(len(tokens_bart)) + "\n\n"
    return final_summary, time, number_of_tokens
    

"""
Summarize text with T5 model
"""
def summarize_chunks_T5(chunks, max_sum_length, min_sum_length, tokenizer, model):
    summaries = []
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer("summarize: "+chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        summary_ids = model.generate(**inputs, max_length=max_sum_length, min_length=min_sum_length, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def summarize_text_T5(long_text, max_text_length=3996):
    start = timer()
    tokenizer_T5= T5Tokenizer.from_pretrained("t5-base", legacy=False) 
    model_T5 = T5ForConditionalGeneration.from_pretrained("t5-base")
    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_T5)
    max_sum_length, min_sum_length = get_max_min_sum_length(num_chunk)
    chunk_summaries_T5 = summarize_chunks_T5(text_chunks, max_sum_length, min_sum_length, tokenizer_T5, model_T5)
    final_summary_T5 = concatenate_summaries(chunk_summaries_T5)
    end = timer()
    time = "Time used: " + str(end - start) + "\n\n"
    tokens_T5 = tokenizer_T5.tokenize(final_summary_T5)
    number_of_tokens = "Number of tokens: " + str(len(tokens_T5)) + "\n\n"
    return final_summary_T5, time, number_of_tokens


"""
Summarize text with LlaMa2 model
"""
def summarize_chunks_llama2(token_chunks, max_sum_length, min_sum_length, tokenizer):
    summaries = []
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        llm = Llama(
            model_path="./../models/llama-2-7b-chat.Q4_K_M.gguf",
            chat_format="llama-2", 
            n_ctx=4096
        )
        completion_llama2 = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": chunk_text},
                {
                    "role": "user",
                    "content": "Please summarize the text to between {} and {} tokens.".format(min_sum_length, max_sum_length)
                }
            ],
            max_tokens=max_sum_length
        )
        message_llama2 = completion_llama2['choices'][0]['message']['content']
        summaries.append(message_llama2)
    return summaries

def summarize_text_llama2(long_text):
    start = timer()
    tokenizer_llama2 = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_llama2, 3996)
    max_sum_length, min_sum_length = get_max_min_sum_length(num_chunk)
    chunk_summaries_llama2 = summarize_chunks_llama2(text_chunks, max_sum_length, min_sum_length, tokenizer_llama2)
    final_summary_llama2 = concatenate_summaries(chunk_summaries_llama2)
    end = timer()
    time = "Time used: " + str(end - start) + "\n\n"
    tokens_llama2 = tokenizer_llama2.tokenize(final_summary_llama2)
    number_of_tokens = "Number of tokens: " + str(len(tokens_llama2)) + "\n\n"
    return final_summary_llama2, time, number_of_tokens


"""
Summarize text with Mistral model
"""
def summarize_chunks_mistral(token_chunks, max_sum_length, min_sum_length, tokenizer):
    summaries = []
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        mistral = Llama(
            model_path="./../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            chat_format="llama-2", 
            n_ctx=4096
        )
        completion_mistral = mistral.create_chat_completion(
            messages = [
                {"role": "system", "content": chunk_text},
                {
                    "role": "user",
                    "content": "Please summarize the text to between {} and {} tokens.".format(min_sum_length, max_sum_length)
                }
            ],
            max_tokens=max_sum_length
        )
        message_mistral = completion_mistral['choices'][0]['message']['content']
        summaries.append(message_mistral)
    return summaries

def summarize_text_mistral(long_text):
    start = timer()
    tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_mistral, 3996)
    max_sum_length, min_sum_length = get_max_min_sum_length(num_chunk)
    chunk_summaries_mistral = summarize_chunks_mistral(text_chunks, max_sum_length, min_sum_length, tokenizer_mistral)
    final_summary_mistral = concatenate_summaries(chunk_summaries_mistral)
    end = timer()
    time = "Time used: " + str(end - start) + "\n\n"
    tokens_mistral = tokenizer_mistral.tokenize(final_summary_mistral)
    number_of_tokens = "Number of tokens: " + str(len(tokens_mistral)) + "\n\n"
    return final_summary_mistral, time, number_of_tokens


"""
Call corresponding summarizer function based on argument.
"""
def summarize_text_get_performance(long_text, gpt_summary, paper_title):
    # bart
    summary_bart, time_bart, number_of_tokens_bart = summarize_text_bart(long_text)
    score_bart = get_score(gpt_summary, summary_bart)
    performance_bart = time_bart + number_of_tokens_bart + score_bart
    with open("output/bart/"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(summary_bart)
    with open("output/bart/performance-"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(performance_bart)
    # t5
    summary_t5, time_t5, number_of_tokens_t5 =  summarize_text_T5(long_text)
    score_t5 = get_score(gpt_summary, summary_t5)
    performance_t5 = time_t5 + number_of_tokens_t5 + score_t5
    with open("output/t5/"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(summary_t5)
    with open("output/t5/performance-"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(performance_t5)
    # llama2
    summary_llama2, time_llama2, number_of_tokens_llama2 =  summarize_text_llama2(long_text)
    score_llama2 = get_score(gpt_summary, summary_llama2)
    performance_llama2 = time_llama2 + number_of_tokens_llama2 + score_llama2
    with open("output/llama2/"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(summary_llama2)
    with open("output/llama2/performance-"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(performance_llama2)
    # mistral
    summary_mistral, time_mistral, number_of_tokens_mistral =  summarize_text_mistral(long_text)
    score_mistral = get_score(gpt_summary, summary_mistral)
    performance_mistral = time_mistral + number_of_tokens_mistral + score_mistral
    with open("output/mistral/"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(summary_mistral)
    with open("output/mistral/performance-"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(performance_mistral)


"""
Calculate evaluation scores
""" 
def rouge_score(references, predictions):
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=[predictions], references=[references])
    return "ROUGE Score: "+ str(rouge_scores) +"\n\n"

def blue_score(references, predictions):
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=[predictions], references=[references])
    return "BLEU Score: " + str(bleu_score) +"\n\n"

def meteor_score(references, predictions):
    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=[predictions], references=[references])
    return "METEOR Score: " + str(meteor_score) +"\n\n"

def get_score(references, predictions):
    score = ""
    score += rouge_score(references, predictions)
    score += blue_score(references, predictions)
    score += meteor_score(references, predictions)
    return score


"""
Main function
"""
def main():
    """Get extracted txt file from upload"""
    txt_path = ask_for_file()
    file = open(txt_path, "r", encoding="utf-8")
    text = file.read()
    file.close()

    """Filter text"""
    long_text = text_filter(text)

    """Read gpt4 summary as reference"""
    paper_title = get_paper_title(txt_path)
    file = open("reference/"+paper_title+".txt", "r", encoding="utf-8")
    gpt_summary = file.read()
    file.close()

    """Summarize text and get performance (time, number of tokens, rouge, blue, meteor scores)"""
    summarize_text_get_performance(long_text, gpt_summary, paper_title)


if __name__ == "__main__":
    main()