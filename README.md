# How to run
## 1. Download LlaMa2 and Mistral GGUF models, and put them in /models folder
LlaMa2: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf

Mistral: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
## 2. Create and activate virtual environment
In Linux terminal: 

`python -m venv venv`

`source venv/bin/activate`
## 3. Install dependencies
`pip install -r requirements.txt`

Install tkinter for file selector:

`sudo apt install python3-tk -y`
## 4. Run the pdf to ppt generator
`python pdf_ppt_generator.py --summarizer=<summarizer> --converter=<converter> (--openaikey=<openaikey>)`

Parameters:
* summarizer: model used for summarizing pdf text. Available options are: `bart`, `t5`, `llama2`, `mistral`
* converter: model used for converting summary into slides content. Available options are: `gpt3`, `llama2`, `mistral`
* openaikey: OpenAI API key to access GPT3 model. Compulsory if `gpt3` is used as the converter.

The output can be found in: **output/summarizer converter/** folder

***

For example, to run the generator with BART summarizer and LlaMa2 converter, the command is:

`python pdf_ppt_generator.py --summarizer=bart --converter=llama2`

And the output will be found in output/bart llama2/ folder
