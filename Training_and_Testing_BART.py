import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Initialize NLP tools and models
reader = easyocr.Reader(['en'])
model_dir = "D:/BTech/sem5/CC-NLP/nlp_model_BART_sciq"
tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)
yolo_model = YOLO('D:/BTech/sem5/CC-NLP/yolov5su.pt')

# Helper functions
def extract_text_from_image(img):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = reader.readtext(img_rgb)
        return " ".join([text for (_, text, _) in results]).strip()
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def summarize_image_content_with_quantities(results):
    try:
        detected_objects = [yolo_model.names[int(box.cls[0].item())] for result in results for box in result.boxes]
        object_counts = Counter(detected_objects)
        return "\n".join([f"- {obj}: {count}" for obj, count in object_counts.items()]) or "No objects detected."
    except Exception as e:
        st.error(f"Error summarizing detected objects: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    try:
        with io.BytesIO(pdf_file.read()) as f:
            return extract_text(f, laparams=LAParams())
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def generate_questions(input_text, num_questions=5, max_length=50):
    try:
        inputs = tokenizer(f"Generate Question: {input_text}", return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_questions,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=2.0,
            no_repeat_ngram_size=2
        )
        return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def calculate_cosine_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] if tfidf_matrix.shape[1] else 0.0

def rank_questions(reference_text, questions):
    return sorted(
        [{'question': question, 'cosine_similarity': calculate_cosine_similarity(reference_text, question)} for question in questions],
        key=lambda x: x['cosine_similarity'],
        reverse=True
    )

def plot_similarity_metrics(questions, cosine_scores):
    if not questions or not cosine_scores:
        st.warning("No questions or scores to plot.")
        return

    st.subheader("Cosine Similarity Scores")
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(questions)), cosine_scores, color='lightblue')
    plt.xticks(range(len(questions)), [f"Q{i+1}" for i in range(len(questions))], rotation=45)
    plt.xlabel("Question Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity Scores for Questions")
    st.pyplot(plt)
    plt.clf()

def main():
    st.title("Question Generation System")
    st.session_state.setdefault('extracted_text', "")

    input_type = st.selectbox("Select Input Type:", ["Select", "Image for OCR", "Image for Object Detection", "Text", "PDF"])

    if input_type == "Image for OCR":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            if st.button("Extract Text"):
                st.session_state.extracted_text = extract_text_from_image(img_cv)
                st.write(st.session_state.extracted_text)

    elif input_type == "Image for Object Detection":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Objects"):
                results = yolo_model(img_cv)
                st.write(summarize_image_content_with_quantities(results))

    elif input_type == "Text":
        input_text = st.text_area("Enter your text here:")
        if st.button("Submit Text"):
            st.session_state.extracted_text = input_text

    elif input_type == "PDF":
        uploaded_file = st.file_uploader("Upload a PDF File", type="pdf")
        if uploaded_file:
            st.session_state.extracted_text = extract_text_from_pdf(uploaded_file)

    if st.session_state.get('extracted_text'):
        st.subheader("Question Generation")
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=3)
        if st.button("Generate Questions"):
            generated_questions = generate_questions(st.session_state.extracted_text, num_questions)
            st.write("Generated Questions:")
            for i, question in enumerate(generated_questions, 1):
                st.write(f"{i}. {question}")

            ranked_questions = rank_questions(st.session_state.extracted_text, generated_questions)
            cosine_scores = [item['cosine_similarity'] for item in ranked_questions]
            plot_similarity_metrics([item['question'] for item in ranked_questions], cosine_scores)

if __name__ == "__main__":
    main()



#python -m streamlit run .\nlp_miniproject_BART.py
import streamlit as st
import cv2
import requests
import numpy as np
import easyocr
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import ngrams
import io

nltk.download('all', quiet=True)

reader = easyocr.Reader(['en'])  

model_dir = "D:/BTech/sem5/CC-NLP/nlp_model_T5"
t5_tokenizer = T5Tokenizer.from_pretrained(model_dir)
t5_model = T5ForConditionalGeneration.from_pretrained(model_dir, from_tf=False, use_safetensors=True)

yolo_model = YOLO('D:/BTech/sem5/CC-NLP/yolov5su.pt')

def extract_text_from_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)
    extracted_text = ""
    for (bbox, text, prob) in results:
        extracted_text += f"{text} "
    return extracted_text.strip()


def summarize_image_content_with_quantities(results):
    detected_objects = []
    for result in results:
        for box in result.boxes:
            detected_class = int(box.cls[0].item())
            class_name = yolo_model.names[detected_class]  
            detected_objects.append(class_name)

    object_counts = Counter(detected_objects)

    if object_counts:
        description = "The image contains the following objects:\n"
        for obj, count in object_counts.items():
            description += f"- {obj}: {count}\n"
    else:
        description = "No objects were detected in the image."

    return description

def extract_text_from_pdf(pdf_file):
    text = extract_text(pdf_file, laparams=LAParams())
    return text

def generate_questions(input_text, num_questions=5, max_length=50):
    input_text = "Generate Question: " + input_text
    inputs = t5_tokenizer(input_text, return_tensors="pt")  
    questions = []
    for _ in range(num_questions):
        outputs = t5_model.generate(  
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=2.0,
            no_repeat_ngram_size=2
        )
        decoded_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        questions.append(decoded_output)
    return questions

def calculate_cosine_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    if tfidf_matrix.shape[1] == 0:
        return 0.0  
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cosine_sim[0][0]

def rank_questions(reference_text, questions):
    scores = []
    for question in questions:
        cosine_score = calculate_cosine_similarity(reference_text, question)
        scores.append({
            'question': question,
            'cosine_similarity': cosine_score
        })
    return sorted(scores, key=lambda x: x['cosine_similarity'], reverse=True)

def plot_cosine_similarity_scores(questions, cosine_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(questions)), cosine_scores, color='lightgreen')
    plt.title("Cosine Similarity Scores for Generated Questions", fontsize=16)
    plt.xlabel("Question Index", fontsize=14)
    plt.ylabel("Cosine Similarity", fontsize=14)
    plt.xticks(np.arange(len(questions)), labels=[f"Q{i+1}" for i in range(len(questions))], rotation=45)
    st.pyplot(plt) 
    plt.clf()  

def plot_pos_tag_frequency(questions):
    pos_tags = []
    for question_text in questions:
        words = word_tokenize(question_text)
        pos_tags.extend([tag for word, tag in nltk.pos_tag(words)])
    
    pos_counts = Counter(pos_tags)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(pos_counts.keys()), y=list(pos_counts.values()), palette='magma')
    plt.title("Part-of-Speech Tag Frequency in Generated Questions", fontsize=16)
    plt.xlabel("POS Tag", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(plt)  
    plt.clf()  

def plot_similarity_heatmap(questions):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm',
                xticklabels=[f"Q{i+1}" for i in range(len(questions))],
                yticklabels=[f"Q{i+1}" for i in range(len(questions))])
    plt.title("Cosine Similarity Heatmap between Questions", fontsize=16)
    plt.xlabel("Questions")
    plt.ylabel("Questions")
    st.pyplot(plt)  
    plt.clf()  

def plot_top_bigrams(questions):
    all_words = ' '.join([word.lower() for question_text in questions for word in word_tokenize(question_text)])
    bigrams = list(ngrams(all_words.split(), 2))

    bigram_freq = Counter(bigrams)
    most_common_bigrams = bigram_freq.most_common(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f'{b[0][0]} {b[0][1]}' for b in most_common_bigrams], y=[b[1] for b in most_common_bigrams], palette='coolwarm')
    plt.title("Top 10 Most Frequent Bi-grams in Generated Questions", fontsize=16)
    plt.xlabel("Bi-gram", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(plt)  
    plt.clf() 

st.title("Question Generation System")

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = "" 

input_type = st.selectbox("Select Input Type:", 
                           ["Select Input Type", 
                            "Image for OCR", 
                            "Image for Object Detection", 
                            "Normal Text", 
                            "PDF"])

if input_type == "Image for OCR":
    st.subheader("Upload an Image for OCR")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Extract Text"):
            st.session_state.extracted_text = extract_text_from_image(img_cv)
            st.subheader("Extracted Text:")
            st.write(st.session_state.extracted_text)

elif input_type == "Image for Object Detection":
    st.subheader("Upload an Image for Object Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            results = yolo_model(img_cv)
            description = summarize_image_content_with_quantities(results)
            st.subheader("Detected Objects:")
            st.write(description)

            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Annotated Image", use_column_width=True)

elif input_type == "Normal Text":
    st.subheader("Enter Your Text")
    input_text = st.text_area("Type your text here...")
    if st.button("Submit Text"):
        st.session_state.extracted_text = input_text
        st.subheader("Input Text:")
        st.write(st.session_state.extracted_text)

elif input_type == "PDF":
    st.subheader("Upload a PDF File")
    uploaded_file = st.file_uploader("Choose a PDF file...", type="pdf")

    if uploaded_file is not None:
        st.session_state.extracted_text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Text from PDF:")
        st.write(st.session_state.extracted_text)


def download_questions(ranked_questions):
    output = io.StringIO()
    for i, question in enumerate(ranked_questions, 1):
        output.write(f"Rank {i}: {question['question']} (Cosine Similarity: {question['cosine_similarity']:.4f})\n")
    
    output.seek(0)
    return output.getvalue()

if st.session_state.extracted_text:
    st.subheader("Question Generation")
    num_questions = st.number_input("Number of questions to generate:", min_value=1, max_value=10, value=3)
    
    if st.button("Generate Questions"):
        generated_questions = generate_questions(st.session_state.extracted_text, num_questions)
        st.subheader("Generated Questions:")
        for i, question in enumerate(generated_questions, 1):
            st.write(f"{i}. {question}")

        ranked_questions = rank_questions(st.session_state.extracted_text, generated_questions)
        
        st.subheader("Ranked Questions:")
        for i, item in enumerate(ranked_questions, 1):
            st.write(f"{i}. {item['question']} (Cosine Similarity: {item['cosine_similarity']:.4f})")

        cosine_scores = [item['cosine_similarity'] for item in ranked_questions]

        plot_cosine_similarity_scores(generated_questions, cosine_scores)

        plot_pos_tag_frequency(generated_questions)

        plot_similarity_heatmap(generated_questions)

        plot_top_bigrams(generated_questions)

        download_data = download_questions(ranked_questions)
        st.download_button(
            label="Download Ranked Questions",
            data=download_data,
            file_name="ranked_questions.txt",
            mime="text/plain"
        )

#python -m streamlit run nlp_miniproject_T5.py