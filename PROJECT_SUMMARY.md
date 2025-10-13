# 🔗 Vector Database RAG - Project Blog

## Problem Statement

As a student constantly working with research papers, textbooks, and documentation—I found myself overwhelmed by the amount of information to process and remember. Traditional search methods weren't cutting it for semantic understanding. I needed a solution that was:
- **Intelligent**: Understand meaning, not just keywords
- **Fast**: Retrieve relevant information instantly
- **Private**: Run entirely locally without sending data to external APIs
- **Scalable**: Handle large documents efficiently

This project became my exploration into building a production-ready RAG (Retrieval-Augmented Generation) system from scratch.

---

## Tech Stack

I wanted to explore modern AI/ML technologies while building something practical:

### Backend
- **Python 3.10+**: Core programming language
- **Qdrant**: High-performance vector database for semantic search
- **Docker**: Containerized database for easy setup and deployment

### AI/ML Stack
- **Sentence-Transformers**: Creating high-quality text embeddings (`all-mpnet-base-v2`)
- **Hugging Face Transformers**: Accessing state-of-the-art language models
- **PyTorch**: Deep learning framework with GPU acceleration support
- **Spacy**: Advanced natural language processing for text chunking

### Document Processing
- **PyMuPDF (fitz)**: Efficient PDF text extraction
- **Rich**: Beautiful terminal UI and progress tracking

### Why Local-First Architecture?
- **Privacy**: All data processing happens on your machine
- **Cost**: No API fees for embeddings or LLM inference
- **Control**: Full control over model selection and parameters
- **Learning**: Understand every component of the system

---

## High Level Design

```mermaid
graph TB
    subgraph "User Interface"
        Terminal[Rich Terminal UI]
        User[User Queries]
    end

    subgraph "Application Layer"
        Pipeline[RAG Pipeline]
        PDFProc[PDF Processor]
        Embedder[Embedding Handler]
        LLM[LLM Handler]
    end

    subgraph "Data Layer"
        Qdrant[(Qdrant Vector DB<br/>Port 6333)]
        VectorStore[Vector Storage]
        Chunks[Text Chunks]
    end

    User --> Terminal
    Terminal --> Pipeline
    Pipeline --> PDFProc
    PDFProc --> Chunks
    Chunks --> Embedder
    Embedder --> VectorStore
    VectorStore --> Qdrant
    
    User --> Pipeline
    Pipeline --> LLM
    Pipeline --> Qdrant
    Qdrant --> LLM

    style Terminal fill:#e1f5ff
    style Pipeline fill:#c8e6c9
    style Qdrant fill:#ffccbc
```

### System Flow

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant PDFProcessor
    participant Embedder
    participant Qdrant
    participant LLM

    User->>Pipeline: Select PDF Document
    Pipeline->>PDFProcessor: Process PDF
    PDFProcessor->>PDFProcessor: Extract & chunk text
    PDFProcessor-->>Pipeline: Return chunks
    
    Pipeline->>Embedder: Create embeddings
    Embedder->>Embedder: Generate vectors
    Embedder-->>Pipeline: Return embeddings
    
    Pipeline->>Qdrant: Store vectors + metadata
    Qdrant-->>Pipeline: Confirm storage
    
    User->>Pipeline: Ask question
    Pipeline->>Embedder: Embed query
    Embedder-->>Pipeline: Query vector
    
    Pipeline->>Qdrant: Semantic search
    Qdrant-->>Pipeline: Top-k relevant chunks
    
    Pipeline->>LLM: Generate response with context
    LLM-->>Pipeline: Generated answer
    Pipeline-->>User: Display response with sources
```

---

## Low Level Design

### Database Schema

```mermaid
erDiagram
    COLLECTIONS ||--o{ VECTORS : contains
    COLLECTIONS {
        string name PK
        int vectors_count
        int vector_size
        string distance_metric
    }
    VECTORS {
        int id PK
        float[] embedding
        string page_number
        text chunk_text
        int chunk_char_count
        int chunk_token_count
    }
```

### Document Processing Flow

```mermaid
stateDiagram-v2
    [*] --> SelectPDF
    SelectPDF --> ExtractText : User selects file
    ExtractText --> SentenceSegmentation : PyMuPDF extraction
    SentenceSegmentation --> ChunkCreation : Spacy NLP
    
    ChunkCreation --> ValidateChunk : Sliding window
    ValidateChunk --> ChunkCreation : Token count OK
    ValidateChunk --> CreateEmbedding : All chunks ready
    
    CreateEmbedding --> StoreVector : Sentence-Transformers
    StoreVector --> [*] : Qdrant upsert
```

### Component Architecture

```mermaid
graph TB
    Main[main.py]
    Main --> Pipeline[SimpleLocalRAG]
    
    Pipeline --> PDFProc[pdf_processor.py]
    Pipeline --> EmbedHand[embedding_handler.py]
    Pipeline --> VectorDB[vector_db_handler.py]
    Pipeline --> LLMHand[llm_handler.py]
    Pipeline --> Utils[utils.py]
    Pipeline --> Config[config.py]
    
    PDFProc --> Spacy[Spacy NLP]
    PDFProc --> PyMuPDF[PyMuPDF/fitz]
    
    EmbedHand --> SentTrans[Sentence-Transformers]
    
    VectorDB --> Qdrant[Qdrant Client]
    
    LLMHand --> HF[Hugging Face Models]
    
    style Main fill:#fff3e0
    style Pipeline fill:#c5e1a5
    style Qdrant fill:#90caf9
```

### API/Module Interactions

```mermaid
graph LR
    subgraph "Core Functions"
        ProcessPDF[process_pdf]
        CreateEmbed[create_embeddings]
        QueryDB[query_qdrant]
        GenerateResp[generate_response]
    end

    subgraph "Helper Functions"
        SelectPDF[select_pdf_path]
        SetupLog[setup_logging]
        CheckCUDA[check_cuda_support]
        CreateColl[create_or_get_collection]
    end

    ProcessPDF --> CreateEmbed
    CreateEmbed --> CreateColl
    CreateColl --> QueryDB
    QueryDB --> GenerateResp
    
    SelectPDF --> ProcessPDF
    SetupLog --> ProcessPDF
    CheckCUDA --> CreateEmbed

    style ProcessPDF fill:#4caf50
    style CreateEmbed fill:#2196f3
    style QueryDB fill:#ff9800
    style GenerateResp fill:#9c27b0
```

---

## Challenges Faced and Overcome

### 1. **First Time with Vector Databases**
**Challenge**: Coming from traditional SQL databases, I struggled with:
- Understanding vector similarity search
- Choosing appropriate distance metrics (cosine vs. euclidean)
- Optimizing embedding dimensions

**Solution**:
- Read Qdrant's documentation thoroughly
- Experimented with different embedding models
- Used cosine similarity for normalized semantic search
- Visualized token distributions to optimize chunk sizes

### 2. **Qdrant Docker Connection Issues**
**Challenge**: Application couldn't connect to Qdrant running in Docker container.

**Error**:
```
ConnectionError: Cannot connect to Qdrant at localhost:6333
```

**Solution**:
- Properly exposed port 6333 in Docker run command
- Increased timeout settings in QdrantClient initialization
- Ensured Docker container was running before application start
- Used volume mounts for data persistence

### 3. **Text Chunking Strategy**
**Challenge**: Fixed-size chunks often split sentences mid-thought, losing semantic coherence.

**Solution**:
- Implemented sliding window chunking with overlap
- Used Spacy for sentence-level segmentation
- Set optimal chunk size: 256 tokens with 30 token overlap
- Visualized token distributions to validate approach

### 4. **GPU Memory Management**
**Challenge**: Large language models exceeded available GPU memory (8GB).

**Solution**:
- Implemented 4-bit quantization using bitsandbytes
- Selected smaller but efficient models (Gemma-2B)
- Added automatic fallback to CPU if GPU unavailable
- Monitored memory usage with PyTorch profiling

### 5. **Embedding Model Selection**
**Challenge**: Needed balance between quality and speed.

**Solution**:
- Benchmarked multiple models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- Chose `all-mpnet-base-v2` for best quality (768 dimensions)
- Implemented batch processing to speed up embedding creation
- Added progress bars for user feedback

### 6. **Context Window Limitations**
**Challenge**: Retrieved chunks exceeded LLM context window, causing truncation.

**Solution**:
- Limited context to 6000 characters
- Implemented smart truncation prioritizing top-scored chunks
- Added token counting before LLM inference
- Designed prompts to work within constraints

### 7. **PDF Extraction Quality**
**Challenge**: Some PDFs had garbled text, tables, or image artifacts.

**Solution**:
- Used PyMuPDF for robust text extraction
- Filtered out low-quality text blocks
- Removed excessive whitespace and special characters
- Added page offset configuration for documents with cover pages

---

## Key Learnings

1. **Vector Embeddings**: Understanding that semantic search is fundamentally different from keyword search—embeddings capture meaning, not just word matches
2. **Chunking Strategy**: Proper text segmentation is critical—too large and you lose precision, too small and you lose context
3. **GPU Utilization**: PyTorch's CUDA support dramatically speeds up both embedding and inference—10x faster than CPU
4. **Model Selection**: Smaller quantized models can provide 80% of the quality at 20% of the memory cost
5. **Docker Compose**: Simplified local development—one command to start the entire stack
6. **Progress Feedback**: Rich library made the terminal experience professional and informative

---

## Future Enhancements

- [ ] Multi-document support (process and query multiple PDFs simultaneously)
- [ ] Web interface using Gradio or Streamlit
- [ ] Support for additional document formats (DOCX, TXT, HTML, Markdown)
- [ ] Hybrid search combining vector similarity and keyword matching
- [ ] Caching mechanism for frequently accessed embeddings
- [ ] Fine-tuning embedding models on domain-specific data
- [ ] Export conversations and sources to PDF reports
- [ ] API endpoint for integration with other applications
- [ ] Real-time document monitoring and automatic re-indexing
- [ ] Multi-language support for non-English documents

---

## Conclusion

This project pushed me deep into the world of modern AI/ML systems. Working with vector databases taught me how semantic search works under the hood, and implementing RAG from scratch gave me insights that using pre-built solutions never would.

The biggest takeaway? **Understanding your data is crucial**—90% of RAG quality comes from proper document processing, chunking, and retrieval strategy. The LLM is just the final step.

If you're building something similar, start with good chunking logic, experiment with different embedding models, and always visualize your data before scaling up. Test each component thoroughly before moving on.

---

## Resources

- **Code Repository**: [GitHub Link](https://github.com/Voodels/PrivateRagFromScratch)
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **Sentence-Transformers**: https://www.sbert.net/
- **Hugging Face**: https://huggingface.co/
- **PyMuPDF**: https://pymupdf.readthedocs.io/

---

*Built with ❤️ by Vighnesh*
