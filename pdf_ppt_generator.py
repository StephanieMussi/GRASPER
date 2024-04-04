
import fitz  
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from transformers import pipeline
from pptx import Presentation

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  
    all_text = ""  

    for page in doc:
        all_text += page.get_text() 

    doc.close() 
    return all_text


def chunk_text(text, max_tokens=1024):
    chunks = []
    for i in range(0, len(text), max_tokens):
        chunks.append(text[i:i+max_tokens])
    return chunks

def summarize_chunks(chunks):
    summaries = []
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs.input_ids, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def concatenate_summaries(summaries):
    return " ".join(summaries)

def get_ppt_content(summary):

    pipe = pipeline(
        "text-generation",
        model="h2oai/h2o-danube-1.8b-chat",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # We use the HF Tokenizer chat template to format each message
    # https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {"role": "user", "content": "Please convert the following text into a presentation. Give title and content for each slide. " +summary},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    res = pipe(
        prompt,
        max_new_tokens=256,
    )
    return res[0]["generated_text"]



def add_slide(prs, title, content):
    slide_layout = prs.slide_layouts[1]  # Use slide layout index 1 for title slide
    slide = prs.slides.add_slide(slide_layout)
    title_placeholder = slide.shapes.title
    content_placeholder = slide.placeholders[1]

    title_placeholder.text = title
    content_placeholder.text = content

def generate_ppt():
    # Create a PowerPoint presentation object
    prs = Presentation()

    # Add slides with titles and content
    slides = [
        ("Introduction", "Paper Title: XCon: Learning with Experts for Fine-grained Category Discovery\n\nProblem Addressed: Generalized Category Discovery (GCD)\n\nMethodology: Expert-Contrastive Learning (XCon)\n\nKey Results: Improved performance over previous methods"),
        ("Background and Motivation", "Challenge: Generic category discovery requires large datasets like ImageNet or COCO, which may not always be feasible.\n\nFormalization of GCD: Leveraging unlabeled data to discover categories, focusing on fine-grained concepts.\n\nLimitation of Existing Approaches: Unsupervised representations may cluster data based on irrelevant cues.\n\nProposed Solution: Expert Contrastive Learning (XCon) to eliminate negative influences and discover fine-grained categories effectively."),
        ("Methodology Overview", "XCon Method: Partition data into k expert sub-datasets using k-means clustering.\n\nEach sub-dataset treated as an expert dataset to eliminate negative influences.\n\nObjective: Learn discriminative features for fine-grained category discovery."),
        ("Contrastive Learning in XCon", "Utilizing k-means grouping on self-supervised features for informative contrastive pairs.\n\nJoint contrastive representation learning on partitioned sub-datasets.\n\nClear performance improvements over previous GCD methods with contrastive learning."),
        ("Representation Learning Challenges", "Challenge: Representations need to be sensitive to detailed discriminative traits.\n\nLeveraging self-supervised representations for rough clustering based on overall image statistics.\n\nProposed approach: Supervised and self-supervised contrastive loss to fine-tune the model."),
        ("Evaluation Metrics", "Splitting training data into labeled (Dl) and unlabeled (Du) datasets.\n\nMeasuring performance using clustering accuracy (ACC) on the unlabeled set."),
        ("Experimental Setup", "Backbone: ViT-B-16\n\nBatch size: 256\n\nTraining epochs: 60 for ImageNet dataset\n\nImplementation: Projection heads as three-layer MLPs"),
        ("Results on Generic Datasets", "Comparison with state-of-the-art methods on CIFAR10, 100, 200, and Stanford Cars.\n\nXCon consistently outperforms baseline methods, demonstrating robust effectiveness."),
        ("Results on Fine-grained Datasets", "Performance improvements on CUB-200 and Stanford Cars benchmarks.\n\nXCon's effectiveness across different α values analyzed."),
        ("Qualitative Analysis", "Visualization of features using t-SNE for qualitative comparison.\n\nClear boundaries between different groups with XCon, corresponding to specific categories."),
        ("Conclusion", "Proposal of XCon for generalized category discovery with self-supervised representation.\n\nImproved performance on image classification benchmarks, validating the method's effectiveness."),
        ("Acknowledgments", "Acknowledgment of compute support from LunarAI."),
        ("References", "Relevant papers and resources cited in the presentation for further reading.")
    ]

    for slide_title, slide_content in slides:
        add_slide(prs, slide_title, slide_content)

    # Save the presentation
    prs.save("presentation.pptx")

def main():
    # Step 1: extract text
    pdf_path = "XCon Learning with Experts for Fine-grained Category Discovery.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)
    # print(pdf_text)

    # Step 2: Summarize 
    text_chunks = chunk_text(pdf_text)
    chunk_summaries = summarize_chunks(text_chunks)
    final_summary = concatenate_summaries(chunk_summaries)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokens = tokenizer.tokenize(final_summary)
    number_of_tokens = len(tokens)
    print("number_of_tokens: ", number_of_tokens)
    print(final_summary)

    # Step 3: Transform into ppt content
    get_ppt_content(final_summary)

    # Step 4: Generate ppt slides
    #TODO: feed result of last step into this function
    generate_ppt()


if __name__ == "__main__":
    main()