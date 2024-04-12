import os
import sys
import fitz  
import copy 
import math
import time
import argparse
import threading
import tkinter as tk
from openai import OpenAI
from llama_cpp import Llama
from pptx import Presentation
from tkinter import filedialog
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LEDTokenizer, LEDForConditionalGeneration

def ask_for_file():
    print("Please upload paper in pdf format:")
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        print(f"File selected: {file_path}")
    else:
        sys.exit("No file selected. Exiting...")
    return file_path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  
    all_text = ""  
    for page in doc:
        all_text += page.get_text() 
    doc.close() 
    return all_text

def get_paper_title(pdf_path):
    filename_with_extension = os.path.basename(pdf_path)
    paper_title, extension = os.path.splitext(filename_with_extension)
    return paper_title

def validate_arguments(args):
    if args.summarizer != 'bart' and args.summarizer != 't5' and args.summarizer != 'led':
        sys.exit("Error: No valid summarizer specified. Please use '--summarizer=bart' or '--summarizer=t5' or '--summarizer=led'.")
    
    if args.converter == 'gpt3':
        if args.openaikey is None:
            sys.exit("Error: No OpenAI API key provided. Please use '--openaikey=<your-openai-key>'.")
    elif args.converter != 'llama2' and  args.converter != 'mistral':
        sys.exit("Error: No valid converter specified. Please use '--converter=gpt3' or '--converter=llama2' or '--converter=mistral'.")

def chunk_text_into_tokens(text, tokenizer, max_tokens=500):
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    count = len(token_chunks)
    return token_chunks, count

def get_max_min_sum_length(max_text_length, num_chunk):
    max_sum_length = min(500, math.floor(max_text_length / num_chunk))
    min_sum_length = math.floor(0.5 * max_sum_length)
    return max_sum_length, min_sum_length

def summarize_chunks(token_chunks, max_sum_length, min_sum_length, tokenizer, model):
    summaries = []
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        summary_ids = model.generate(**inputs, max_length=max_sum_length, min_length=min_sum_length, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def summarize_chunks_T5(chunks, max_sum_length, min_sum_length, tokenizer, model):
    summaries = []
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer("summarize: "+chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        summary_ids = model.generate(**inputs, max_length=max_sum_length, min_length=min_sum_length, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def concatenate_summaries(summaries):
    return " ".join(summaries)


def summarize_text_bart(long_text, max_text_length):
    tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_bart)
    max_sum_length, min_sum_length = get_max_min_sum_length(max_text_length, num_chunk)
    chunk_summaries = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_bart, model_bart)
    final_summary = concatenate_summaries(chunk_summaries)
    return final_summary

def summarize_text_T5(long_text, max_text_length):
    tokenizer_T5= T5Tokenizer.from_pretrained("t5-base", legacy=False) 
    model_T5 = T5ForConditionalGeneration.from_pretrained("t5-base")
    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_T5)
    max_sum_length, min_sum_length = get_max_min_sum_length(max_text_length, num_chunk)
    chunk_summaries_T5 = summarize_chunks_T5(text_chunks, max_sum_length, min_sum_length, tokenizer_T5, model_T5)
    final_summary_T5 = concatenate_summaries(chunk_summaries_T5)
    return final_summary_T5

def summarize_text_LED(long_text, max_text_length):
    tokenizer_LED = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model_LED = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
    text_chunks, num_chunk = chunk_text_into_tokens(long_text, tokenizer_LED)
    max_sum_length, min_sum_length = get_max_min_sum_length(max_text_length, num_chunk)
    chunk_summaries_LED = summarize_chunks(text_chunks, max_sum_length, min_sum_length, tokenizer_LED, model_LED)
    final_summary_LED = concatenate_summaries(chunk_summaries_LED)
    return final_summary_LED

def summarize_text(long_text, summarizer_type):
    max_text_length = 4096
    if summarizer_type == 'bart':
        return summarize_text_bart(long_text, max_text_length)
    elif summarizer_type == 't5':
        return summarize_text_t5(long_text, max_text_length)
    elif summarizer_type == 'led':
        return summarize_text_led(long_text, max_text_length)

def convert_content_gpt3(final_summary, openaikey):
    client = OpenAI(api_key = openaikey)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": final_summary},
        {"role": "user", "content": "Please convert the text into a presentation slides. Give the title and actual detailed content for each slide, in the format of \"Slide 1\nTitle: (title)\nContent: (content)\". Do not add other information."}
    ]
    )
    message = completion.choices[0].message.content
    return message

def convert_content_llama2(final_summary):
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        llm = Llama(
            model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
            chat_format="llama-2", 
            n_ctx=4096
        )
        completion_llama2 = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": final_summary},
                {
                    "role": "user",
                    "content": "Please convert the text into presentation slides. Give the title and actual detailed content for each slide, in the format of 'Slide 1\nTitle: (title)\nContent: (content)'. Do not add other information."
                }
            ]
        )
        message_llama2 = completion_llama2['choices'][0]['message']['content']
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    return message_llama2

def convert_content_mistral(final_summary):
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        mistral = Llama(
            model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            chat_format="llama-2", 
            n_ctx=4096
        )
        completion_mistral = mistral.create_chat_completion(
            messages = [
                {"role": "system", "content": final_summary},
                {
                    "role": "user",
                    "content": "Please convert the text into a presentation slides. Give the title and actual detailed content for each slide, in the format of \"Slide 1\nTitle: (title)\nContent: (content)\". Do not add other information. "
                }
            ]
        )
        message_mistral = completion_mistral['choices'][0]['message']['content']
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    return message_mistral

def convert_content(final_summary, converter_type, openaikey=None):
    if converter_type == 'gpt3':
        return convert_content_gpt3(final_summary, openaikey)
    elif converter_type == 'llama2':
        return convert_content_llama2(final_summary)
    elif converter_type == 'mistral':
        return convert_content_mistral(final_summary)

def animate():
    global summarizing
    i = 0
    while summarizing:
        sys.stdout.write("\rSummarizing text" + "." * i + " " * (3 - i)) 
        sys.stdout.flush()
        time.sleep(0.5)  
        i = (i + 1) % 4 
    sys.stdout.write("\r" + " " * 20 + "\r")  
    sys.stdout.flush()

def animate_conversion():
    global converting
    i = 0
    while converting:
        sys.stdout.write("\rConverting content" + "." * i + " " * (3 - i))  
        sys.stdout.flush()
        time.sleep(0.5) 
        i = (i + 1) % 4  
    sys.stdout.write("\r" + " " * 30 + "\r") 
    sys.stdout.flush()

def add_title_slide(title, prs):
    first_slide = prs.slides[0]
    title_shape = first_slide.shapes[0]
    if title_shape.has_text_frame:
        p = title_shape.text_frame.paragraphs[0]  
        run = p.runs[0] if p.runs else p.add_run()  
        run.text = title 
        
def add_content_slide(title, content, prs):
    first_slide = prs.slides[-1]
    title_shape = first_slide.shapes[0]
    if title_shape.has_text_frame:
        p = title_shape.text_frame.paragraphs[0]  
        run = p.runs[0] if p.runs else p.add_run()  
        run.text = title 
    content_shape = first_slide.shapes[1]
    if content_shape.has_text_frame:
        p = content_shape.text_frame.paragraphs[0]  
        run = p.runs[0] if p.runs else p.add_run()  
        run.text = content 

def duplicate_slide(presentation, slide_index):
    slide = presentation.slides[slide_index]
    layout = slide.slide_layout
    duplicated_slide = presentation.slides.add_slide(layout)
    for shape in slide.shapes:
        el = shape.element
        new_el = copy.deepcopy(el)
        duplicated_slide.shapes._spTree.insert_element_before(new_el, 'p:extLst')
    return presentation

def generate_slides(paper_title, message, template_path, save_path):
    slides = message.strip().split("Slide ")[1:]
    prs = Presentation(template_path)
    add_title_slide(paper_title, prs)
    for i, slide in enumerate(slides):
        parts = slide.split("\nTitle:")
        slide_number = parts[0].strip()  
        title_content = parts[1].split("\nContent:")
        title = title_content[0].strip()
        content = title_content[1].strip()
        if i > 0:
            prs = duplicate_slide(prs, -1)
        add_content_slide(title, content, prs)
        
    prs.save(save_path)

def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--summarizer', type=str, help='Choose the model summarizer (BART or T5 or LED)')
    parser.add_argument('--converter', type=str, help='Choose the model converter (gpt3 or llama2 or mistral2)')
    parser.add_argument('--openaikey', type=str, help='Provide your OpenAI API key to access GPT3 API')
    args = parser.parse_args()
    validate_arguments(args)

    """Get pdf file from upload"""
    pdf_path = ask_for_file()

    """Step1. Extract text from pdf"""
    print("Extracting text...")
    long_text = extract_text_from_pdf(pdf_path)
    paper_title = get_paper_title(pdf_path)

    """Step2. Summarize text to reduce length"""
    global summarizing
    summarizing = True
    anim_thread = threading.Thread(target=animate)
    anim_thread.start()
    final_summary = summarize_text(long_text, args.summarizer)
    summarizing = False
    anim_thread.join() 

    """Step3. Convert summary into slides content"""
    global converting
    converting = True
    conv_thread = threading.Thread(target=animate_conversion)
    conv_thread.start()
    message = convert_content(final_summary, args.converter, args.openaikey)
    converting = False
    conv_thread.join() 

    """Step4. Generate presentation slides from content"""
    print("Generating slides...")
    template_path = './template.pptx'
    save_path = './presentation.pptx'
    generate_slides(paper_title, message, template_path, save_path)
    print("Slides is saved to: " + os.path.abspath(save_path))


if __name__ == "__main__":
    main()