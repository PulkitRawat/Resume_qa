import os 
import pdfplumber
import pytesseract
import re
import spacy
import torch
import chromadb
import uuid
import numpy as np

from PIL import Image
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
chroma_client = chromadb.PersistentClient(path = "./chroma_db")


def extract_text(file_path:str)->str:
    """
    extract text from file(PDF or Image)
    Parameters:
        file_path(str): Path to file. Supported formats:  "pdf", "jpg", "jpeg", "png".
    Returns:
        str: extracted text form the file.
    Raises:
        ValueError: when the file type is unsupported
    """
    text = ""
    _,file_extension = os.path.splitext(file_path)
    file_extension= file_extension.lower()
    if file_extension in [".pdf"]:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text+= page.extract_text()
        except Exception as e:
            return f"Error in parsing the pdf: {e}"
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            return f"Error in parsing in image: {e}"
    else:
        raise ValueError("unsupported file type. Please use 'pdf', 'jpg', 'jpeg' or 'png'")
    return text.strip()

def clean_text(text:str)->str:
    """
    Cleans the extracted text including:
     -Removing extra spaces and new Lines.
     -Handling common ocr errors (e.g., 'ﬁ' to 'fi').
     -Normalizing punctuation.
    Parameters:
        text(str): Text to be cleaned.
    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\s+', ' ', text) # replace multiple spaces with single space
    text = re.sub(r'[^\x00-\x7F]+', '', text) # remove non-ASCII characters
    text = re.sub(r'ﬁ', 'fi', text) # common ocr mistake
    return text
def segment_into_sentences(text: str)->list[str]:
    """
    Segments cleaned text into individual sentences. the sentences are the stream of words in single line
    Parameters:
        text(str): Cleaned text to be segmented
    Returns:
        list[str]: list of sentences extracted from the text.
    """
    sentence_endings = re.compile(r'(\.|\n|\t)')
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def preProcess_chunk_text(text: str, chunk_size=10) -> list[list[str]]:
    """
    Divide the sentences from the cleaned text into manageable chunks. Each chunk contains sentences, and the number of words is minimum the chunk_size.

    Parameters:
        text (str): Parsed text obtained from the file that is to be chunked.
        chunk_size (int): Maximum number of words per chunk (Default: 20).

    Returns:
        list[list[str]]: List of chunks, where each chunk contains sentences with a word count greater than or equal to chunk_size.
    """
    text = clean_text(text)
    sentences = segment_into_sentences(text)

    chunks = []
    chunk = []
    word_count = 0
    
    for sentence in sentences:
        if(sentence == "."):
            continue
        chunk.append(sentence)
        word_count += len(sentence.split())
        if(word_count>chunk_size):
            chunks.append(' '.join(chunk))
            chunk = []
            word_count = 0
            
    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

model_path = "BERT_FineTuned_Model2"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

def extract_keywords(text: str):
    """
    extracts the key phases and noun from the question.
    Parameters:
        text(str): text form which keywords will be extracted.
        
    Returns: 
        List[str]: a list of key phrases.
    """
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ','VERB']]
    return keywords
    
def get_sentence_embedding(text:str):
    """
    Convert a sentence into its dense embedding usng RoBERTa.
    Parameters:
        text(str): the text to be converted into embedding.
        
    Returns:
        embedding: the dense vector representing embedding.
    """
    inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    embedding = hidden_states.mean(dim = 1).squeeze()
    return embedding

def store_embeddings(text_chunks:list[str], collection):
    """
    stores text chunks and their embeddings into chroma DB.

    Parameters:
        text_chunks(list[str]): list of extracted text chunks.
        collection: place to store embedding
    Returns:
        None(Stores the embedding in chromaDB)
    """
    
    
    existing_texts = set()  

    for chunk in text_chunks:
        if chunk in existing_texts:
            continue  
        
        embedding = get_sentence_embedding(chunk).tolist()
        
        collection.add(
            ids=[str(uuid.uuid4())],  
            embeddings=[embedding],  
            metadatas=[{"text": chunk}]
        )
        
        existing_texts.add(chunk) 

    
def get_most_relevant_sentences(question: str, collection, top_k = 7)->list[str]:
    """
    find the most relevant sentences in the resume for a given question using chromaDB and boost the relevance using keyword matching

    Parameters: 
        questions(str): the user's query.
        collection: place to store embedding.
        top_k: maximum number of relvant sentences required(default 7)
    Returns:
        List[str]: the list of most relevant chunks
    """
    keywords = extract_keywords(question)
    question_embedding = get_sentence_embedding(question).numpy()
    results = collection.query(
        query_embeddings = [question_embedding],
        n_results = int(top_k*5)
    )

    retrieved_chunks = results['metadatas'][0]
    distances = results['distances'][0]

    text_chunks = [chunk['text'] for chunk in retrieved_chunks]
    keyword_boost = []
    for chunk,distance in zip(text_chunks, distances):
    
        chunk_keywords = extract_keywords(chunk)

        match_count = len([chunk_keyword for chunk_keyword in chunk_keywords if chunk_keyword in keywords])
        
        sim = 1/(1+distance)
        boosted_score= sim + match_count*0.54
        keyword_boost.append((boosted_score, chunk))
        
    keyword_boost.sort(key = lambda x:x[0], reverse = True)
    
    top_k_chunks = [chunk for _, chunk in keyword_boost[:top_k]]

    return top_k_chunks

qa_model_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_path)
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_path)

def generate_answer(question: str, relevant_sentences:list[str])->str:
    """
    Generate a clear and concise answer to the question based on the most relevant sentences.

    Parameters:
        question(str):The user's query.
        relevant_sentences(list[str]): list of most relevant sentences.
        
    Returns:
        str: Generate answer.
    """

    context = " ".join(relevant_sentences)
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors='pt', truncation = True, max_length = 512)
    # print(inputs)
    with torch.no_grad():
        output = qa_model(**inputs)
        start_scores,end_scores, = output.start_logits, output.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1

    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))

    return answer.strip()

def find_answer(question: str, text: str):
    """
    Final function to generate answer that will be given to the flask 

    Parameters:
        question (str): query asked by the user.
        text (str): text given by the user in form of pdf or image.
        
    Returns:
        str: generated answer.
    """
    
    resume_chunks = preProcess_chunk_text(text) 
    existing_collections = chroma_client.list_collections()
    if "resume_embeddings" in [col.name for col in existing_collections]:
        chroma_client.delete_collection(name="resume_embeddings")
    
    collection = chroma_client.get_or_create_collection(name="resume_embeddings")
    store_embeddings(resume_chunks, collection)
    top_relevant_sentences = get_most_relevant_sentences(question, collection, 10)
    answer = generate_answer(question, top_relevant_sentences)
    return answer