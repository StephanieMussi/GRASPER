import os
import re
import math
import torch
from transformers import BartTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration
from transformers import AutoTokenizer, ProphetNetForConditionalGeneration
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, GemmaForCausalLM
import os
import evaluate


max_text_length = 4096

def text_filter(text):
    # 删除Abstract前的内容
    abstract_index = text.find("Abstract")
    long_text = long_text[abstract_index:]
    # 删除参考文献标题及其后的内容
    references_index = long_text.find("References")
    long_text = long_text[:references_index]
    # 移除参考文献编号
    long_text = re.sub(r'\[\d+\]', '', long_text)
    # 移除表格
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

def chunk_text_into_tokens(text, tokenizer, max_tokens=500):
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    count = len(token_chunks)
    return token_chunks, count

def summarize_chunks(token_chunks, max_sum_length, min_sum_length, tokenizer, model):
    summaries = []

    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move inputs to device
        summary_ids = model.generate(**inputs, max_length=max_sum_length, min_length=min_sum_length, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries

def concatenate_summaries(summaries):
    return " ".join(summaries)

def Bart(long_text):
    tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_bart)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_bart = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_bart, model_bart)

    final_summary_bart = concatenate_summaries(chunk_summaries_bart)

    tokens_bart = tokenizer_bart.tokenize(final_summary_bart)
    number_of_tokens_bart = len(tokens_bart)
    print("number_of_tokens: ", number_of_tokens_bart)

    print(final_summary_bart)
    with open("summary_bart.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_bart)

def summarize_chunks_T5(chunks, max_sum_length, min_sum_length, tokenizer, model):
    summaries = []

    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer("summarize: "+chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        summary_ids = model.generate(**inputs, max_length=max_sum_length, min_length=min_sum_length, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries

def T5(long_text):
    tokenizer_T5 = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    model_T5 = T5ForConditionalGeneration.from_pretrained("t5-base")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_T5)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_T5 = summarize_chunks_T5(text_chunks, max_sum_length, min_sum_length, tokenizer_T5, model_T5)

    final_summary_T5 = concatenate_summaries(chunk_summaries_T5)

    tokens_T5 = tokenizer_T5.tokenize(final_summary_T5)
    number_of_tokens_T5 = len(tokens_T5)
    print("number_of_tokens: ", number_of_tokens_T5)

    print(final_summary_T5)
    with open("summary_t5.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_T5)

def LED(long_text):
    tokenizer_LED = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model_LED = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_LED)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_LED = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_LED, model_LED)

    final_summary_LED = concatenate_summaries(chunk_summaries_LED)

    tokens_LED = tokenizer_LED.tokenize(final_summary_LED)
    number_of_tokens_LED = len(tokens_LED)
    print("number_of_tokens: ", number_of_tokens_LED)

    print(final_summary_LED)
    with open("summary_led.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_LED)

def Pegasus(long_text):
    tokenizer_Pegasus = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model_Pegasus = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_Pegasus)

    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_Pegasus = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_Pegasus,
                                               model_Pegasus)

    final_summary_Pegasus = concatenate_summaries(chunk_summaries_Pegasus)

    tokens_Pegasus = tokenizer_Pegasus.tokenize(final_summary_Pegasus)
    number_of_tokens_Pegasus = len(tokens_Pegasus)
    print("number_of_tokens: ", number_of_tokens_Pegasus)

    print(final_summary_Pegasus)
    with open("summary_pegasus.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_Pegasus)

def BigBirdPegasus(long_text):
    tokenizer_BigBird = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
    model_BigBird = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_BigBird)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_BigBird = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_BigBird,
                                               model_BigBird)

    final_summary_BigBird = concatenate_summaries(chunk_summaries_BigBird)

    tokens_BigBird = tokenizer_BigBird.tokenize(final_summary_BigBird)
    number_of_tokens_BigBird = len(tokens_BigBird)
    print("number_of_tokens: ", number_of_tokens_BigBird)

    print(final_summary_BigBird)
    with open("summary_bigbird.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_BigBird)

def ProphetNet(long_text):
    tokenizer_prophetNet = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
    model_prophetNet = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_prophetNet)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_prophetNet = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_prophetNet,
                                                  model_prophetNet)

    final_summary_prophetNet = concatenate_summaries(chunk_summaries_prophetNet)

    tokens_prophetNet = tokenizer_prophetNet.tokenize(final_summary_prophetNet)
    number_of_tokens_prophetNet = len(tokens_prophetNet)
    print("number_of_tokens: ", number_of_tokens_prophetNet)

    print(final_summary_prophetNet)
    with open("summary_prophetnet.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_prophetNet)

def LlaMa2(long_text):
    tokenizer_llama2 = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model_llama2 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_llama2)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_llama2 = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_llama2,
                                              model_llama2)

    final_summary_llama2 = concatenate_summaries(chunk_summaries_llama2)

    tokens_llama2 = tokenizer_llama2.tokenize(final_summary_llama2)
    number_of_tokens_llama2 = len(tokens_llama2)
    print("number_of_tokens: ", number_of_tokens_llama2)

    print(final_summary_llama2)
    with open("summary_llama2.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_mistral)

def Mistral(long_text):
    tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model_mistral = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_mistral)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_mistral = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_mistral,
                                               model_mistral)

    final_summary_mistral = concatenate_summaries(chunk_summaries_mistral)

    tokens_mistral = tokenizer_mistral.tokenize(final_summary_mistral)
    number_of_tokens_mistral = len(tokens_mistral)
    print("number_of_tokens: ", number_of_tokens_mistral)

    print(final_summary_mistral)
    with open("summary_mistral.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_mistral)

def Gemma(long_text):
    model_gemma = GemmaForCausalLM.from_pretrained("google/gemma-7b")
    tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-7b")

    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_gemma)
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)

    chunk_summaries_gemma = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_gemma, model_gemma)

    final_summary_gemma = concatenate_summaries(chunk_summaries_gemma)

    tokens_gemma = tokenizer_gemma.tokenize(final_summary_gemma)
    number_of_tokens_gemma = len(tokens_gemma)
    print("number_of_tokens: ", number_of_tokens_gemma)

    print(final_summary_gemma)
    with open("summary_gemma.txt", "w", encoding="utf-8") as file:
        file.write(final_summary_gemma)

def summarize_text(long_text, summarizer_type):
    if summarizer_type == 'bart':
        Bart(long_text)
    elif summarizer_type == 't5':
        T5(long_text)
    elif summarizer_type == 'led':
        LED(long_text)
    elif summarizer_type == 'pegasus':
        Pegasus(long_text)
    elif summarizer_type == 'bigbird':
        BigBirdPegasus(long_text)
    elif summarizer_type == 'prophetnet':
        ProphetNet(long_text)
    elif summarizer_type == 'llama2':
        LlaMa2(long_text)
    elif summarizer_type == 'mistral':
        Mistral(long_text)
    elif summarizer_type == 'gemma':
        Gemma(long_text)

def rouge_score(references,file_path):
    rouge = evaluate.load("rouge")
    file = open(file_path, "r", encoding="utf-8")
    predictions = file.read()
    file.close()
    rouge_scores = rouge.compute(predictions=[predictions], references=[references])
    print("ROUGE Score:", rouge_scores)

def rouge_score(references,file_path):
    rouge = evaluate.load("rouge")
    file = open(file_path, "r", encoding="utf-8")
    predictions = file.read()
    file.close()
    rouge_scores = rouge.compute(predictions=[predictions], references=[references])
    print("ROUGE Score:", rouge_scores)

def blue_score(references,file_path):
    bleu = evaluate.load("bleu")
    file = open(file_path, "r", encoding="utf-8")
    predictions = file.read()
    file.close()
    bleu_score = bleu.compute(predictions=[predictions], references=[references])
    print("BLEU Score:", bleu_score)

def meteor_score(references,file_path):
    meteor = evaluate.load("meteor")
    file = open(file_path, "r", encoding="utf-8")
    predictions = file.read()
    file.close()
    meteor_score = meteor.compute(predictions=[predictions], references=[references])
    print("METEOR Score:", meteor_score)

def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--summarizer', type=str, help='Choose the model summarizer (bart/t5/led/pegasus/bigbird/prophetnet/llama2/mistral/gemma )')
    args = parser.parse_args()
    validate_arguments(args)


    file = open("extracted_text.txt", "r", encoding="utf-8")
    text = file.read()
    file.close()

    long_text=text_filter(text)

    summarize_text(long_text, args.summarizer)

    file_path='summary'+args.summarizer+'.txt'

    file = open("gpt_summary.txt", "r", encoding="utf-8")
    gpt_summary = file.read()
    file.close()
    references = gpt_summary
    rouge_score(references,file_path)
    blue_score(references,file_path)
    meteor_score(references,file_path)

if __name__ == "__main__":
    main()