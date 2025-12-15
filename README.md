# 💊 Drug Information Chatbot

A Deep Learning chatbot powered by Flan-T5 + LoRA fine-tuning and Retrieval-Augmented Generation (RAG) for answering drug-related questions.

## 🌟 Features

- **Fine-tuned Language Model**: Flan-T5-Base with LoRA adapters trained on medical Q&A data
- **RAG (Retrieval-Augmented Generation)**: FAISS-based vector search for accurate drug information
- **Interactive UI**: Built with Streamlit for easy interaction
- **Source Attribution**: Shows sources used for each answer
- **Medical Database**: Information from DailyMed and MedQuAD datasets

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)
- Git (for cloning the repository)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd deep-learning-final-project
```

### 2. Create a Virtual Environment (Recommended)

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have a CUDA-enabled GPU and want to use GPU acceleration:

1. Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/)
2. Replace `faiss-cpu` with `faiss-gpu` in requirements.txt

## 📁 Project Structure

```
deep-learning-final-project/
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   └── __init__.py
├── notebooks/
│   ├── faiss_rag_builder.py      # RAG retriever implementation
│   ├── finetuning_model.ipynb    # Model fine-tuning notebook
│   └── ...
├── data/
│   ├── faiss/                     # FAISS index and metadata
│   ├── finetuning/                # Training data
│   └── rag/                       # RAG knowledge base
├── models/
│   └── drug_qna_lora/
│       └── final/                 # Fine-tuned model checkpoint
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## ▶️ Running the Application

### Prerequisites Before Running

Make sure you have:

1. ✅ Trained model files in `models/drug_qna_lora/final/`
2. ✅ FAISS index files in `data/faiss/`
   - `drug_knowledge.index`
   - `metadata.pkl`
   - `config.json`

If these files are missing, you'll need to:

- Run the fine-tuning notebook: `notebooks/finetuning_model.ipynb`
- Build the RAG index: `notebooks/faiss_rag_builder.ipynb`

### Start the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Alternative: Specify Port

```bash
streamlit run app/streamlit_app.py --server.port 8080
```

## 🎯 Using the Chatbot

1. **Ask Questions**: Type your drug-related questions in the chat input
2. **View Answers**: The chatbot will provide answers based on the trained model
3. **Check Sources**: Expand the "Sources Used" section to see the retrieved documents
4. **Try Examples**: Click the example questions in the sidebar
5. **Toggle RAG**: Enable/disable RAG in the settings (sidebar)
6. **Clear Chat**: Use the "Clear Chat" button to start a new conversation

### Example Questions

- "What is the dosage of Amoxicillin?"
- "What are the warnings and precautions of Atorvastatin?"
- "How does Albuterol work?"
- "What are the side effects of Ibuprofen?"
- "When should I not take Amoxicillin?"

## ⚙️ Configuration

### Model Settings

Edit the paths in [app/streamlit_app.py](app/streamlit_app.py):

```python
MODEL_DIR = ROOT / "models" / "drug_qna_lora" / "final"
FAISS_DIR = ROOT / "data" / "faiss"
```

### Generation Parameters

Modify the generation parameters in the `generate_answer()` function:

```python
outputs = model.generate(
    **inputs,
    max_length=512,        # Maximum output length
    num_beams=4,           # Beam search width
    repetition_penalty=1.2, # Avoid repetition
    no_repeat_ngram_size=3, # N-gram blocking
    early_stopping=True,
)
```

## 🛠️ Troubleshooting

### Issue: Model Not Loading

**Error:** "❌ Model not loaded"

**Solution:**

- Ensure you have trained the model first
- Check that the model files exist in `models/drug_qna_lora/final/`
- Run the fine-tuning notebook to create the model

### Issue: RAG Not Available

**Error:** "⚠️ RAG not available"

**Solution:**

- Check that FAISS index files exist in `data/faiss/`
- Run the `faiss_rag_builder.ipynb` notebook to build the index
- Verify the `faiss-cpu` package is installed

### Issue: Out of Memory

**Solution:**

- Reduce batch size or max_length in generation
- Use CPU instead of GPU (set `device_map="cpu"`)
- Close other memory-intensive applications

### Issue: Slow Performance

**Solution:**

- Use GPU if available (install CUDA-enabled PyTorch)
- Reduce `num_beams` parameter
- Cache the model loading (already implemented with `@st.cache_resource`)

## 📊 Training Your Own Model

To train the model from scratch:

1. Prepare your dataset in `data/finetuning/`
2. Open and run `notebooks/finetuning_model.ipynb`
3. The trained model will be saved to `models/drug_qna_lora/final/`

## 🗂️ Building the RAG Index

To rebuild the FAISS index:

1. Ensure your knowledge base is in `data/rag/knowledge_base.json`
2. Open and run `notebooks/faiss_rag_builder.ipynb`
3. The index will be saved to `data/faiss/`

## Important Disclaimer

**This chatbot is for educational purposes only.**

- Do NOT use this as a substitute for professional medical advice
- Always consult with qualified healthcare professionals for medical decisions
- The information provided may not be complete or up-to-date
- This is a student project for a Deep Learning course

## Project Information

- **Course**: Deep Learning Final Project
- **Technology Stack**:
  - Flan-T5-Base (Google)
  - LoRA (Low-Rank Adaptation)
  - FAISS (Facebook AI Similarity Search)
  - Sentence Transformers
  - Streamlit
- **Data Sources**:
  - DailyMed (drug information)
  - MedQuAD (medical Q&A dataset)

## 📝 License

This project is for educational purposes. Please check the licenses of the underlying models and datasets:

- [Flan-T5 License](https://github.com/google-research/t5x)
- [DailyMed Terms](https://dailymed.nlm.nih.gov/dailymed/app-support-web-pages.cfm)
