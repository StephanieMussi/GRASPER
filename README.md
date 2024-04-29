# PDF PPT Generator

## How to run
### Method 1: Run with docker (recommended)
#### 1. Pull the docker image

`docker pull symu/pdf-ppt-generator`
#### 2. Allow access to screen for display

`xhost local:root`
#### 3. Run the container

`docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name pdf-ppt-generator symu/pdf-ppt-generator bash`
#### 4. Run the pdf ppt generator inside the container
`python3 pdf_ppt_generator.py --summarizer=<summarizer> --converter=<converter> (--openaikey=<openaikey>)`

Parameters:
* summarizer: model used for summarizing pdf text. Available options are: `bart`, `t5`, `llama2`, `mistral`
* converter: model used for converting summary into slides content. Available options are: `gpt3`, `llama2`, `mistral`
* openaikey: OpenAI API key to access GPT3 model. Compulsory if `gpt3` is used as the converter.

The output can be found in: **output/summarizer converter/** folder

For example, to run the generator with BART summarizer and LlaMa2 converter, the command is:

`python3 pdf_ppt_generator.py --summarizer=bart --converter=llama2`

And the output will be found in output/bart llama2/ folder

***

### Method 2: Run with virtualenv
#### 1. Clone this repository

`git clone git@github.com:StephanieMussi/pdf_ppt_generator.git`

`cd pdf_ppt_generator`

#### 2. Download LlaMa2 and Mistral GGUF models

`sudo apt update && sudo apt install wget`

`mkdir -p /models`

`wget -O /models/llama-2-7b-chat.Q4_K_M.gguf "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf?raw=true"`

`wget -O /models/mistral-7b-instruct-v0.2.Q4_K_M.gguf "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf?raw=true"`

#### 3. Create and activate virtual environment

`python3 -m venv venv`

`source venv/bin/activate`
#### 4. Install dependencies
`pip install -r requirements.txt`

`sudo apt-get update && sudo apt-get install -y gcc g++ python3-tk`
#### 5. Run the pdf ppt generator
`python3 pdf_ppt_generator.py --summarizer=<summarizer> --converter=<converter> (--openaikey=<openaikey>)`

Parameters:
* summarizer: model used for summarizing pdf text. Available options are: `bart`, `t5`, `llama2`, `mistral`
* converter: model used for converting summary into slides content. Available options are: `gpt3`, `llama2`, `mistral`
* openaikey: OpenAI API key to access GPT3 model. Compulsory if `gpt3` is used as the converter.

The output can be found in: **output/summarizer converter/** folder
