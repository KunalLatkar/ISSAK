# 🧠 Automatic Question Generation using Fine-Tuned T5 & BART

## 📌 Project Overview

This project focuses on building an Automatic Question Generation (AQG)
system using fine-tuned Transformer-based models --- **T5 (Text-to-Text
Transfer Transformer)** and **BART (Bidirectional and Auto-Regressive
Transformers)**.

The system accepts comprehension passages (Text, PDF, or OCR input) and
generates high-quality, context-aware questions. A React frontend is
integrated with a backend inference pipeline for real-time question
generation.

------------------------------------------------------------------------

## 🏗️ System Architecture

### 🔹 High-Level Workflow

![High Level Data Flow Diagram](.DataFlowDiagram.png)

The system pipeline:

1.  User provides Text / PDF / OCR input\
2.  Text extraction and preprocessing\
3.  Transformer-based question generation\
4.  Relevance and fluency scoring\
5.  Ranked downloadable question file

------------------------------------------------------------------------

### 🔹 Detailed Automatic Question Generation Architecture

![Object Diagram](.Object_diagram.drawio.png)

#### Modules:

### 🧑‍💻 User Interface

-   Text Input\
-   PDF Parser\
-   OCR Support

### 🧹 Basic Preprocessing

-   Tokenization\
-   POS Tagging\
-   Named Entity Recognition (NER)

### 🧠 Advanced Processing

-   Dependency Parsing\
-   Question Type Classification\
-   Template-Based Generation\
-   Rule-Based Generation

### 🤖 Transformer-Based Neural Generation

-   Fine-tuned T5\
-   Fine-tuned BART\
-   Context-aware sequence-to-sequence modeling

### 📊 Output Module

-   Question Ranking\
-   Importance Ranking\
-   Post-processing\
-   Output Formatting

### 🔁 Feedback Loop

-   User Feedback Collector\
-   Model Tuner

------------------------------------------------------------------------

## 🧩 Model Architecture Explanation

### 🌟 T5 -- The Text-to-Text Transformer

T5 treats every NLP task as a text-to-text problem.\
It uses a Transformer Encoder-Decoder architecture:

-   Encoder → Understands the comprehension passage\
-   Decoder → Generates the question

It was pre-trained on massive datasets using span corruption objectives,
allowing it to deeply understand contextual relationships. When
fine-tuned for question generation, it adapts efficiently to generate
grammatically correct and semantically meaningful questions.

------------------------------------------------------------------------

### 🌟 BART -- The Denoising Autoencoder

BART combines: - Bidirectional encoder (like BERT)\
- Autoregressive decoder (like GPT)

It was pre-trained by corrupting text and learning to reconstruct it.
This makes it powerful for generation tasks like summarization and
question generation.

For our system: - Encoder builds deep contextual representation\
- Decoder generates fluent questions sequentially

------------------------------------------------------------------------

## ⚙️ Training Configuration

The models were fine-tuned using HuggingFace Transformers with the
following configuration:

### 🔍 Training Details

-   30 epochs for sufficient adaptation\
-   Batch size 8 (balanced for GPU memory)\
-   Frequent logging for monitoring\
-   Custom column handling to avoid dataset mismatch

------------------------------------------------------------------------

## 🖥️ Frontend

![Frontend](.Frontend.png)

A basic React frontend was built to:

-   Accept comprehension input\
-   Send requests to backend\
-   Write and save notes for future reference
-   Display generated questions dynamically


------------------------------------------------------------------------

## 📜 Conclusion

This project demonstrates how transfer learning with large pre-trained
Transformer models can be effectively used to build scalable and
high-quality Question Generation systems.

By combining strong NLP foundations, transformer fine-tuning, ranking
mechanisms, and frontend integration, this system represents a complete
end-to-end deep learning application.
