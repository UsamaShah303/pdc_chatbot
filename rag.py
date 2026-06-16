"""
RAG Pipeline for FinAI Nexus Chatbot
Uses Groq API for LLM generation (free, fast, works on HF Spaces)
Parallel & Distributed Computing CCP
Student: Syed Usama Ali Shah | ID: 61585
"""

import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# ── Configuration ─────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PDF_FOLDER = "docs"

# Groq API — free, fast, works from HuggingFace Spaces
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"  # Free LLaMA on Groq

SYSTEM_PROMPT = """You are FinAI Nexus, an expert AI financial advisor specializing in Pakistani investments.
You provide accurate, helpful information about PSX stocks, mutual funds, risk profiling,
portfolio optimization, and Islamic finance. Always give practical, actionable advice
suitable for Pakistani retail investors. Keep answers clear and concise.
Use the provided context to answer questions accurately."""

# ── Fallback Knowledge Base ────────────────────────────────────────
FALLBACK_KNOWLEDGE = [
    {"text": "PSX stands for Pakistan Stock Exchange headquartered in Karachi. Formed in 2016 through merger of KSE, LSE, and ISE. Lists over 500 companies regulated by SECP. KSE-100 index tracks top 100 companies by market capitalization.", "source": "PSX Investor Guide"},
    {"text": "To invest in PSX: 1) Choose TREC holder broker registered with PSX. 2) Submit CNIC and bank details. 3) Open CDC sub-account. 4) Complete KYC verification. 5) Fund account and start trading. Popular brokers: Arif Habib, JS Global, Foundation Securities.", "source": "PSX Trading Guide"},
    {"text": "Mutual funds in Pakistan regulated by SECP, managed by AMCs. MUFAP represents all AMCs. Types: equity funds, income funds, money market funds, balanced funds, Islamic funds. Minimum investment from PKR 500.", "source": "MUFAP Investor Guide"},
    {"text": "Risk profiling determines investor risk capacity. Categories: Conservative (low risk), Moderate (balanced), Aggressive (high risk). FinAI Nexus uses K-Means clustering on quiz answers to determine risk profile.", "source": "Investment Risk Guide"},
    {"text": "Portfolio diversification spreads investments across sectors to reduce risk. In Pakistan: diversify across PSX sectors (banking, cement, textile, energy), mutual funds, government securities, gold.", "source": "Portfolio Management Guide"},
    {"text": "Islamic finance complies with Shariah law. Prohibits Riba (interest), Gharar (uncertainty). Options: Islamic mutual funds, Sukuk bonds, Shariah-screened PSX stocks. Regulated by SBP in Pakistan.", "source": "Islamic Finance Guide"},
    {"text": "Zakat requires 2.5% payment on qualifying wealth above Nisab (87.48g gold). Applies to stocks, mutual fund units, savings held for one lunar year. Pakistan deducts automatically on 1st Ramadan unless CZ-50 filed.", "source": "Zakat Investment Guide"},
    {"text": "FinAI Nexus is AI-powered robo-advisor for Pakistani investors. Features: risk profiling (K-Means), portfolio optimization (MPT), PSX sentiment analysis (FinBERT), Monte Carlo simulation, AI chatbot.", "source": "FinAI Nexus Documentation"},
    {"text": "Sharpe ratio = (Return - Risk Free Rate) / Standard Deviation. Higher means better risk-adjusted return. Pakistan risk-free rate is 3-month T-bill rate. FinAI Nexus maximizes Sharpe ratio.", "source": "Financial Metrics Guide"},
    {"text": "Start investing with PKR 500 via money market mutual funds on Easypaisa/JazzCash. Pakistan Savings Certificates minimum PKR 500. Regular Investment Plans from PKR 1000/month from AMCs.", "source": "Beginner Investment Guide"},
    {"text": "Financial statements: Balance sheet shows assets and liabilities. Income statement shows revenue and profit. Cash flow statement shows money movement. Key ratios: P/E ratio, EPS, ROE, debt-to-equity.", "source": "Guide on Financial Statements"},
    {"text": "Shareholders rights include: voting at AGM, receiving dividends, inspecting company records, transferring shares, receiving annual reports, participating in rights issues and bonus shares.", "source": "Guide on Shareholders Rights"},
    {"text": "KSE-100 tracks top 100 PSX companies by free-float market cap. Most followed Pakistani stock market indicator. Benchmark for equity mutual fund performance comparison.", "source": "PSX Index Guide"},
    {"text": "SECP regulates capital markets, mutual funds, insurance, NBFIs in Pakistan. All investment products must be SECP-registered. Investor protection is primary mandate.", "source": "SECP Guide"},
    {"text": "SBP is Pakistan central bank managing monetary policy, banking regulation, forex reserves. Policy rate influences borrowing costs and investment returns across all asset classes.", "source": "SBP Guide"},
    {"text": "Modern Portfolio Theory by Markowitz maximizes return for given risk. Uses asset correlations to find efficient frontier. FinAI Nexus implements MPT via PyPortfolioOpt library.", "source": "MPT Guide"},
    {"text": "Monte Carlo simulation runs thousands of random scenarios to forecast portfolio outcomes. FinAI Nexus uses 10000 scenarios showing best case, worst case, expected returns over 1-5 years.", "source": "Monte Carlo Guide"},
    {"text": "FinBERT classifies financial news as Positive, Negative, or Neutral. FinAI Nexus uses FinBERT to analyze PSX company headlines and adjust portfolio weights based on sentiment.", "source": "FinAI Nexus AI Documentation"},
    {"text": "CDC (Central Depository Company) holds shares electronically in Pakistan. Investors need CDC account to hold PSX shares. Eliminates physical share certificates and related risks.", "source": "PSX Trading Guide"},
    {"text": "NCCPL (National Clearing Company) handles settlement of PSX trades. T+2 settlement means trades settle 2 days after execution. Ensures smooth transfer of shares and funds.", "source": "PSX Guide"},
]


def load_pdfs_from_folder(folder_path):
    """Load and chunk PDFs — PARALLEL COMPUTING aspect"""
    chunks = []

    if not os.path.exists(folder_path):
        print(f"⚠️ docs/ folder not found — using fallback knowledge base")
        return []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("⚠️ No PDFs found in docs/ folder")
        return []

    print(f"Found {len(pdf_files)} PDFs: {pdf_files}")

    try:
        import pypdf
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            source_name = pdf_file.replace('.pdf', '').replace('-', ' ').replace('_', ' ')

            try:
                print(f"Loading: {pdf_file}")
                reader = pypdf.PdfReader(pdf_path)
                full_text = ""

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + " "

                # Chunk the text
                words = full_text.split()
                chunk_words = 100
                overlap_words = 10

                for i in range(0, len(words), chunk_words - overlap_words):
                    chunk_text = " ".join(words[i:i + chunk_words])
                    if len(chunk_text.strip()) > 50:
                        chunks.append({
                            "text": chunk_text,
                            "source": source_name,
                        })

                print(f"✅ {pdf_file}: {len(reader.pages)} pages, chunks added")

            except Exception as e:
                print(f"⚠️ Error loading {pdf_file}: {e}")

    except ImportError:
        print("⚠️ pypdf not installed")
        return []

    print(f"✅ Total PDF chunks: {len(chunks)}")
    return chunks


class FinAIRagChatbot:

    def __init__(self):
        print("Initializing FinAI Nexus RAG Chatbot...")

        # Load embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Load knowledge base
        print("Loading knowledge base...")
        pdf_chunks = load_pdfs_from_folder(PDF_FOLDER)

        if pdf_chunks:
            self.knowledge_base = pdf_chunks + FALLBACK_KNOWLEDGE
            print(f"✅ PDF + fallback: {len(self.knowledge_base)} total chunks")
        else:
            self.knowledge_base = FALLBACK_KNOWLEDGE
            print(f"✅ Fallback only: {len(self.knowledge_base)} chunks")

        # Build FAISS index
        print("Building FAISS vector index...")
        self._build_index()

        # Setup Groq
        self._setup_groq()

        print("✅ FinAI Nexus RAG Chatbot ready!")

    def _build_index(self):
        texts = [chunk["text"] for chunk in self.knowledge_base]
        print(f"Embedding {len(texts)} chunks in parallel batches...")

        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        print(f"✅ FAISS index: {self.index.ntotal} vectors, {dimension} dims")

    def _setup_groq(self):
        if GROQ_API_KEY:
            self.headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            self.use_groq = True
            print("✅ Groq API configured — using LLaMA 3.1 8B!")
        else:
            self.use_groq = False
            print("⚠️ No GROQ_API_KEY found — add in Space Settings → Secrets")

    def _retrieve(self, query, top_k=TOP_K):
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True
        ).astype(np.float32)

        distances, indices = self.index.search(query_embedding, top_k)

        retrieved = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.knowledge_base):
                chunk = self.knowledge_base[idx]
                retrieved.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "distance": float(dist),
                })
        return retrieved

    def _generate_via_groq(self, query, context):
        """Call Groq API with retry on rate limit"""
        import time

        # Trim context to reduce token usage
        context_trimmed = context[:800]

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are FinAI Nexus, an expert AI financial advisor for Pakistani investors. Give clear, concise answers based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context: {context_trimmed}\n\nQuestion: {query}\n\nAnswer concisely for Pakistani investors:"
                }
            ],
            "max_tokens": 250,
            "temperature": 0.7,
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    GROQ_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:
                    # Rate limited — wait and retry
                    wait_time = (attempt + 1) * 5
                    print(f"Rate limited — waiting {wait_time}s before retry {attempt+1}/3")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Groq API error: {response.status_code}")
                    return None

            except Exception as e:
                print(f"Groq API failed: {e}")
                return None

        return None

    def _generate_fallback(self, context):
        return f"""Based on Pakistani financial documents:

{context}

---
*Source: Official Pakistani financial guides*"""

    def answer(self, query):
        # Step 1: Retrieve — FAISS parallel search
        retrieved_chunks = self._retrieve(query)

        # Step 2: Build context
        context = "\n\n".join([
            f"[{chunk['source']}]: {chunk['text']}"
            for chunk in retrieved_chunks
        ])

        # Step 3: Generate via Groq or fallback
        if self.use_groq:
            answer = self._generate_via_groq(query, context)
            if not answer:
                answer = self._generate_fallback(context)
        else:
            answer = self._generate_fallback(context)

        sources = list(dict.fromkeys([c["source"] for c in retrieved_chunks]))
        return {"answer": answer, "sources": sources}