import os
import re
import sys
import math
import torch
import evaluate
import argparse
import tkinter as tk
from openai import OpenAI
from llama_cpp import Llama
from tkinter import filedialog
from timeit import default_timer as timer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Validate the arguments specified by user. 
If invalid arguments are given, terminate the program.
"""
def validate_arguments(args):
    if args.openaikey is None:
        sys.exit("Error: No OpenAI API key provided. Please use '--openaikey=<your-openai-key>'.")
    

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
Get paper title from pdf file path.
It is assumed that name of the pdf file is the paper title.
"""
def get_paper_title(pdf_path):
    filename_with_extension = os.path.basename(pdf_path)
    paper_title, extension = os.path.splitext(filename_with_extension)
    return paper_title


"""
Convert summary to slides content with GPT3
"""
def convert_content_gpt3(final_summary, openaikey):
    start = timer()
    client = OpenAI(api_key = openaikey)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": final_summary},
        {"role": "user", "content": "Please convert the text into a presentation slides. Give the title and actual detailed bullet point content for each slide, in the format of \"Slide 1\nTitle: (title)\nContent: (content)\". Do not add other information."}
    ]
    )
    message = completion.choices[0].message.content
    end = timer()
    time = "Time used: " + str(end - start) + "\n\n"
    number_of_words = "Number of words: " + str(len(message.split())) + "\n\n"
    return message, time, number_of_words


"""
Convert summary to slides content with LlaMa2
"""
def convert_content_llama2(final_summary):
    start = timer()
    llm = Llama(
        model_path="./../models/llama-2-7b-chat.Q4_K_M.gguf",
        chat_format="llama-2", 
        n_ctx=4096
    )
    completion_llama2 = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": final_summary},
            {
                "role": "user",
                "content": "Please convert the text into presentation slides. Give the title and actual detailed bullet point content for each slide, in the format of 'Slide 1\nTitle: (title)\nContent: (content)'. Do not add other information."
            }
        ]
    )
    message_llama2 = completion_llama2['choices'][0]['message']['content']
    end = timer()
    time = "Time used: " + str(end - start) + "\n\n"
    number_of_words = "Number of words: " + str(len(message_llama2.split())) + "\n\n"
    return message_llama2, time, number_of_words



"""
Convert summary to slides content with Mistral
"""
def convert_content_mistral(final_summary):
    start = timer()
    mistral = Llama(
        model_path="./../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        chat_format="llama-2", 
        n_ctx=4096
    )
    completion_mistral = mistral.create_chat_completion(
        messages = [
            {"role": "system", "content": final_summary},
            {
                "role": "user",
                "content": "Please convert the text into a presentation slides. Give the title and actual detailed bullet point content for each slide, in the format of \"Slide 1\nTitle: (title)\nContent: (content)\". Do not add other information. "
            }
        ]
    )
    message_mistral = completion_mistral['choices'][0]['message']['content']
    end = timer()
    time = "Time used: " + str(end - start) + "\n\n"
    number_of_words = "Number of words: " + str(len(message_mistral.split())) + "\n\n"
    return message_mistral, time, number_of_words



"""
Call corresponding converter function
"""
def convert_content_get_performance(final_summary, openaikey, gpt_slides_content, paper_title):
    # gpt3
    content_gpt3, time_gpt3, number_of_words_gpt3 = convert_content_gpt3(final_summary, openaikey)
    score_gpt3 = get_score(gpt_slides_content, content_gpt3)
    performance_gpt3 = time_gpt3 + number_of_words_gpt3 + score_gpt3
    with open("output/gpt3/"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(content_gpt3)
    with open("output/gpt3/performance-"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(performance_gpt3)
    # llama2
    content_llama2, time_llama2, number_of_words_llama2 = convert_content_llama2(final_summary)
    score_llama2 = get_score(gpt_slides_content, content_llama2)
    performance_llama2 = time_llama2 + number_of_words_llama2 + score_llama2
    with open("output/llama2/"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(content_llama2)
    with open("output/llama2/performance-"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(performance_llama2)
    # mistral
    content_mistral, time_mistral, number_of_words_mistral = convert_content_mistral(final_summary)
    score_mistral = get_score(gpt_slides_content, content_mistral)
    performance_mistral = time_mistral + number_of_words_mistral + score_mistral
    with open("output/mistral/"+paper_title+".txt", "w", encoding="utf-8") as file:
        file.write(content_mistral)
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
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--openaikey', type=str, help='Provide your OpenAI API key to access GPT3 API')
    args = parser.parse_args()
    validate_arguments(args)

    """Get summary txt file from upload"""
    txt_path = ask_for_file()
    file = open(txt_path, "r", encoding="utf-8")
    text = file.read()
    file.close()

    """Read gpt4 slides content as reference"""
    paper_title = get_paper_title(txt_path)
    file = open("reference/"+paper_title+".txt", "r", encoding="utf-8")
    gpt_slides_content = file.read()
    file.close()

    """Convert slides content and get performance (time, number of words, rouge, blue, meteor scores)"""
    convert_content_get_performance(text, args.openaikey, gpt_slides_content, paper_title)


if __name__ == "__main__":
    main()