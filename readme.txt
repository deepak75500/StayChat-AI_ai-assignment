# The Taj Mahal Palace AI Concierge

A production-style Retrieval-Augmented Generation (RAG) chatbot designed for guests of **The Taj Mahal Palace, Mumbai**. The system answers questions about hotel amenities, rooms, dining, wellness services, policies, and hotel history using a curated knowledge base and semantic retrieval.

## Features

* FAISS-based semantic search
* Retrieval-Augmented Generation (RAG)
* Multi-turn conversation support
* Intent classification
* English, Hindi, and Hinglish support
* Anti-hallucination guardrails
* Evaluation harness with factual and safety tests
* Streamlit-based luxury concierge interface

---

## Problem Statement

Traditional hotel chatbots often provide generic responses and may hallucinate information such as pricing or booking links. The objective of this project is to create a grounded AI concierge that answers only from verified hotel knowledge while preventing unsafe or inaccurate responses.

---

## Technology Stack

| Component       | Technology                             |
| --------------- | -------------------------------------- |
| Frontend        | Streamlit                              |
| Vector Database | FAISS                                  |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2 |
| LLM             | Groq Llama 3.3 70B Versatile           |
| Language        | Python                                 |
| Retrieval       | Semantic Similarity Search             |

---

## Repository Structure

```text
taj-concierge/
│
├── app.py
├── taj_mahal_palace_mumbai_kb.txt
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

---

# Model Architecture and Approach

The system follows a Retrieval-Augmented Generation (RAG) architecture.

## Workflow

```text
Knowledge Base
      │
      ▼
Chunking
(350 words + 50 overlap)
      │
      ▼
Sentence Transformer
(all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Index
      │
      ▼
User Query
      │
      ▼
Intent Classification
      │
      ▼
Query Embedding
      │
      ▼
Top-K Retrieval
      │
      ▼
Prompt Construction
(Context + History)
      │
      ▼
Groq Llama 3.3 70B
      │
      ▼
Guardrail Validation
      │
      ▼
Final Response
```

---

## Knowledge Base Processing

The hotel knowledge base is stored in a plain text file.

### Chunking Strategy

* Chunk Size: 350 words
* Overlap: 50 words
* Sliding window approach
* Preserves contextual continuity

### Why Chunking?

Chunking prevents context loss and improves retrieval accuracy by ensuring relevant information remains available even when facts span multiple sections.

---

## Embedding Model

Model Used:

```python
sentence-transformers/all-MiniLM-L6-v2
```

### Advantages

* Fast inference
* Small memory footprint
* Strong semantic understanding
* 384-dimensional embeddings

Each chunk is converted into a vector representation before being stored inside FAISS.

---

## Vector Search (FAISS)

The system uses:

```python
faiss.IndexFlatIP
```

with L2-normalized embeddings.

### Retrieval Process

1. User query is embedded.
2. Cosine similarity search is performed.
3. Top 5 most relevant chunks are retrieved.
4. Retrieved context is sent to the LLM.

---

## Intent Classification

Before retrieval, every user query is classified into one of the following categories:

| Intent           | Example                       |
| ---------------- | ----------------------------- |
| booking_inquiry  | book room, reservation        |
| amenity_question | spa, gym, pool                |
| complaint        | issue, problem                |
| staff_command    | send towel, arrange transport |
| other            | fallback category             |

Intent classification improves routing and analytics.

---

## Multi-Turn Conversation Memory

The chatbot maintains the last six conversation turns.

This enables:

* Follow-up questions
* Context retention
* Natural conversation flow

Example:

User: Tell me about the Tata Suite.

User: How large is it?

The chatbot understands that "it" refers to the Tata Suite.

---

## Guardrail System

A critical requirement is preventing hallucinated prices and payment links.

### Layer 1: Prompt Constraints

The system prompt explicitly instructs the model:

* Never invent room prices.
* Never generate payment links.
* Only answer using retrieved context.

### Layer 2: Response Validation

Generated responses are scanned using regex patterns.

Blocked examples:

```text
₹25,000 per night
USD 500
Click here to pay
payment link
book now at...
```

If unsafe content is detected, the response is replaced with a safe fallback directing users to the reservations team.

---

## Language Support

Supported Languages:

* English
* Hindi
* Hinglish

The model automatically replies in the user's language.

Example:

User: Pool kab khulta hai?

Response: Swimming pool subah 6:00 AM se khulta hai.

---

## Evaluation Framework

The project includes a built-in evaluation suite.

### Factual Tests

Questions whose answers exist in the knowledge base.

Examples:

* Check-in time
* Tata Suite details
* Spa timings
* Hotel history

### Safety Tests

Questions designed to trigger hallucinations.

Examples:

* Room pricing
* Payment links
* Live ratings

The chatbot must refuse these requests safely.

### Metrics

* Accuracy Score
* Keyword Match Score
* Guardrail Success Rate
* Response Latency

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/taj-concierge-rag.git
cd taj-concierge-rag
```

### Create Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Future Improvements

* Persistent FAISS index storage
* Voice-enabled concierge
* Multi-property hotel support
* Hotel CRM integration
* Feedback collection and analytics
* Advanced LLM-based intent classification

---

## Key Design Decisions

### Why RAG Instead of Fine-Tuning?

RAG ensures:

* Knowledge updates without retraining
* Better factual grounding
* Lower operational cost

### Why FAISS?

* Fast similarity search
* Lightweight deployment
* Industry-standard vector retrieval

### Why MiniLM?

* Fast embedding generation
* Low memory usage
* Strong semantic performance

### Why Groq?

* Extremely low latency inference
* Easy API integration
* No local GPU requirement

---

## Conclusion

The Taj Mahal Palace AI Concierge demonstrates a practical enterprise-grade RAG system that combines semantic retrieval, large language models, conversational memory, and safety guardrails to provide accurate and trustworthy hotel assistance. The architecture prioritizes factual grounding, multilingual accessibility, and hallucination prevention, making it suitable as a foundation for real-world hospitality AI solutions.
