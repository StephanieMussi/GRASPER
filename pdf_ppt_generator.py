import os
import sys
import fitz  
import copy 
import math
import argparse
import tkinter as tk
from openai import OpenAI
from llama_cpp import Llama
from pptx import Presentation
from tkinter import filedialog
from transformers import BartTokenizer, BartForConditionalGeneration

def ask_for_file():
    print("Please upload paper in pdf format:")
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        print(f"File selected: {file_path}")
    else:
        print("No file selected.")
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
    if args.converter == 'gpt3':
        if args.openaikey is None:
            sys.exit("Error: No OpenAI API key provided. Please use '--openaikey=<your-openai-key>'.")
    elif args.converter != 'llama2' and  args.converter != 'mistral':
        sys.exit("Error: No valid converter specified. Please use '--converter=gpt3' or '--converter=llama2' or '--converter=mistral'.")

def chunk_text(text, max_tokens=1024):
    chunks = []
    count = 0
    for i in range(0, len(text), max_tokens):
        chunks.append(text[i:i+max_tokens])
        count = count + 1
    return chunks, count

def summarize_chunks(chunks, max_sum_length):
    summaries = []
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs.input_ids, max_length=max_sum_length, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def concatenate_summaries(summaries):
    return " ".join(summaries)


def summarize_text(long_text, max_text_length):
    text_chunks, num_chunk = chunk_text(long_text)
    max_sum_length = math.floor(max_text_length / num_chunk)
    chunk_summaries = summarize_chunks(text_chunks, max_sum_length)
    final_summary = concatenate_summaries(chunk_summaries)
    return final_summary

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
    llm = Llama(
        model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
        chat_format="llama-2", 
        n_ctx=4096
    )
    completion_llama2 = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": final_summary},
            {
                "role": "user",
                "content": "Please convert the text into a presentation slides. Give the title and actual detailed content for each slide, in the format of \"Slide 1\nTitle: (title)\nContent: (content)\". Do not add other information. "
            }
        ]
    )
    message_llama2 = completion_llama2['choices'][0]['message']['content']
    return message_llama2

def convert_content_mistral(final_summary):
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
    return message_mistral

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
    print("Summarizing text...")
    max_text_length = 4096
    final_summary = summarize_text(long_text, max_text_length)

    """Step3. Convert summary into slides content"""
    print("Converting content...")
    if args.converter == 'gpt3':
        print("Using GPT-3 model")
        message = convert_content_gpt3(final_summary, args.openaikey)
    elif args.converter == 'llama2':
        print("Using LLaMA-2 model")
        message = convert_content_llama2(final_summary)
    elif args.converter == 'mistral':
        print("Using Mistral model")
        message = convert_content_mistral(final_summary)

    """Step4. Generate presentation slides from content"""
    print("Generating slides...")
    template_path = './template.pptx'
    save_path = './presentation.pptx'
    generate_slides(paper_title, message, template_path, save_path)
    print("Slides is saved to: " + os.path.abspath(save_path))


if __name__ == "__main__":
    main()