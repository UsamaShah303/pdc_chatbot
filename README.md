---
title: FinAI Nexus RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: true
license: mit
short_description: AI Financial Advisor for Pakistani Investors
---

# FinAI Nexus RAG Chatbot

**AI-powered financial advisory chatbot for Pakistani investors**

## About

This chatbot combines two powerful AI techniques:

- **RAG (Retrieval Augmented Generation)** — retrieves relevant financial documents before answering
- **Fine-tuned LLaMA 3.2 3B** — generates accurate answers using QLoRA fine-tuning on Pakistani financial Q&A

## Features

- PSX stock market questions
- Mutual fund guidance (MUFAP)
- Risk profiling explanation
- Islamic finance information
- Portfolio diversification advice
- FinAI Nexus platform questions

## Tech Stack

| Component | Technology |
|---|---|
| Base LLM | LLaMA 3.2 3B |
| Fine-tuning | QLoRA (4-bit quantization) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| RAG Framework | Custom Python pipeline |
| Frontend | Gradio 5 |
| Hosting | HuggingFace Spaces |

## Academic Context

**Course:** Parallel and Distributed Computing — CCP

**Student:** Syed Usama Ali Shah | ID: 61585

**Institution:** Iqra University Karachi — BS Computer Science 2022-2026

## Parallel & Distributed Computing Aspects

- Batch GPU embedding (parallel vector computation)
- FAISS multi-threaded similarity search
- QLoRA distributed fine-tuning with gradient checkpointing
- Mixed precision (fp16) parallel matrix operations
