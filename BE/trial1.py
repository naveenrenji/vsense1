import spacy
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import fitz  
import numpy as np

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF for more robust handling."""
    text = ''
    try:
        pdf = fitz.open(pdf_path)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def preprocess_text(text):
    """Splits the text into sentences without altering the tokens."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def get_embeddings(sentences):
    """Generates embeddings for a list of sentences."""
    return model.encode(sentences)

# def assign_tags_based_on_similarity(sentences, sentence_embeddings):
#     """Assigns predefined tags to sentences based on semantic similarity."""
#     tags = ["Workforce Demographics", "Women in the Workforce", "Migrant Workers",
#             "Employment Shifts and Categories", "Unemployment Trends"]
#     tag_embeddings = model.encode(tags)
    
#     tag_assignments = []
#     for sentence_embedding in sentence_embeddings:
#         similarities = util.pytorch_cos_sim(sentence_embedding, tag_embeddings)[0].cpu().numpy()
#         highest_similarity_index = np.argmax(similarities)
#         tag_assignments.append(tags[highest_similarity_index])
    
#     return tag_assignments

def assign_tags_based_on_similarity(sentences, sentence_embeddings, threshold=0.5):
    """Assigns predefined tags to sentences based on semantic similarity exceeding a threshold."""
    tags = ["Workforce Demographics", "Women in the Workforce", "Migrant Workers",
            "Employment Shifts and Categories", "Unemployment Trends"]
    tag_embeddings = model.encode(tags)
    
    sentence_tag_assignments = []
    for sentence_embedding in sentence_embeddings:
        similarities = util.pytorch_cos_sim(sentence_embedding, tag_embeddings)[0].cpu().numpy()
        
        assigned_tags = [tags[idx] for idx, similarity in enumerate(similarities) if similarity > threshold]
        
        if not assigned_tags:
            assigned_tags.append('Other')
        
        sentence_tag_assignments.append(assigned_tags)
    
    return sentence_tag_assignments



def main(pdf_path, output_csv_path='tagged_sentences.csv'):
    text = extract_text_from_pdf(pdf_path)
    if text:
        sentences = preprocess_text(text)
        sentence_embeddings = get_embeddings(sentences)
        assigned_tags = assign_tags_based_on_similarity(sentences, sentence_embeddings)

        # Group sentences by their assigned tags
        tagged_sentences = pd.DataFrame({
            'Tag': assigned_tags,
            'Sentences': sentences
        })
        
        tagged_sentences.to_csv(output_csv_path, index=False)
        print(f"Output saved to {output_csv_path}")
    else:
        print("No text extracted from the PDF.")

if __name__ == "__main__":
    pdf_path = 'sample_reports/sampleresearchreport1.pdf' 
    main(pdf_path)
