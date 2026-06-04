# YouTube Comment Analyzer GitHub Projects (Advanced Analysis)

A curated compilation of advanced YouTube comment analysis projects sourced from GitHub, categorized by their primary analytical techniques (such as Deep Learning, Large Language Models, Multi-Agent workflows, and Real-Time Stream processing).

---

## 🚀 Top Advanced YouTube Comment Analyzers

The following table summarizes the most sophisticated open-source repositories designed for advanced comment analysis:

| Repository | Primary Tech Stack | Core Advanced Features | Link |
| :--- | :--- | :--- | :--- |
| **nolancacheux/AI-Video-Comment-Analyzer** | Next.js 15, FastAPI, BERT, BERTopic, Ollama | BERT sentiment detection, BERTopic semantic clustering, and local LLM summarization. | [View Repo](https://github.com/nolancacheux/AI-Video-Comment-Analyzer) |
| **SercanTeyhani/youtube-comment-analyzer-chatbot** | Streamlit, LDA, BERTopic, LangChain, Google Gen AI | Topic modeling, trend tracking, and an interactive chatbot for Q&A on comments. | [View Repo](https://github.com/SercanTeyhani/youtube-comment-analyzer-chatbot) |
| **AdhamOudeif/google-gen-ai-capstone** | Python, RAG, Embeddings, Agentic LLMs | Retrieval-Augmented Generation (RAG) and semantic vector search on comment databases. | [View Repo](https://github.com/AdhamOudeif/google-gen-ai-capstone) |
| **emirhansilsupur/youtube-video-analyzer** | CrewAI, Python, PDF Export | Multi-agent collaborative inspection (CrewAI) for video comments and PDF report generation. | [View Repo](https://github.com/emirhansilsupur/youtube-video-analyzer) |
| **C0HEr/YTComments** | Apache Kafka, Apache Spark, Streamlit | Big data real-time ingestion and stream analysis utilizing Spark and Kafka. | [View Repo](https://github.com/C0HEr/YTComments) |
| **mohd-musheer/youtube-comments-sentiment-analysis** | Python, Streamlit, ML Classifiers | Analyzes comments to construct a "mental health/sentiment profile" of the video community. | [View Repo](https://github.com/mohd-musheer/youtube-comments-sentiment-analysis) |
| **baladitya445/llm-based-live-yt-comments-analyzer** | Python, LLMs, Creator Dashboard | LLM-based live chat/comment metrics and creator-specific insights. | [View Repo](https://github.com/baladitya445/llm-based-live-yt-comments-analyzer) |

---

## 🔍 In-Depth Repository Breakdown

### 1. `nolancacheux/AI-Video-Comment-Analyzer`
* **Architecture**: Modern Full-Stack (Next.js 15 Frontend + FastAPI Backend).
* **NLP Suite**:
  * **BERT**: Performs high-accuracy sentiment detection.
  * **BERTopic**: Performs unsupervised semantic topic clustering to identify core conversational themes.
  * **Ollama integration**: Summarizes comments locally without relying on expensive cloud API keys.
* **Why it's advanced**: Combining neural sentiment networks, neural topic modeling (BERTopic), and local LLMs provides a production-grade analytic capability.

### 2. `SercanTeyhani/youtube-comment-analyzer-chatbot`
* **Architecture**: Streamlit UI with LangChain integration.
* **Key Capabilities**:
  * **Interactive Chatbot**: Allows creators to ask questions directly about their comments (e.g., *"What did viewers think of the introduction scene?"*).
  * **Topic Modeling**: Integrates both classical LDA (Latent Dirichlet Allocation) and neural BERTopic.
  * **Historical Trends**: Visualizes sentiment shifts across upload histories.
* **Why it's advanced**: Shifts the paradigm from passive data visualization to active conversational intelligence via RAG/LangChain.

### 3. `AdhamOudeif/google-gen-ai-capstone`
* **Architecture**: Capstone-grade project using Google Gen AI frameworks.
* **Key Capabilities**:
  * **Agentic Workflows**: Employs agentic LLMs to reason over scraped comment logs.
  * **Embedding and Vectorization**: Encodes comments into a high-dimensional vector space.
  * **RAG Pipeline**: Resolves complex natural language queries by searching vectorized comment stores.
* **Why it's advanced**: Applies cutting-edge Agentic RAG concepts specifically to user feedback loops at scale.

### 4. `emirhansilsupur/youtube-video-analyzer`
* **Architecture**: Multi-agent framework powered by CrewAI.
* **Key Capabilities**:
  * **Collaborative AI Agents**: Assigns distinct tasks (comment fetching, sentiment analysis, reporting) to autonomous agents.
  * **Comprehensive Reporting**: Automatically designs and exports high-quality PDF performance summaries.
* **Why it's advanced**: Uses multi-agent orchestration to divide-and-conquer YouTube API ingestion and descriptive reporting.

### 5. `C0HEr/YTComments`
* **Architecture**: Distributed Big Data Pipeline (Kafka + Spark + Streamlit).
* **Key Capabilities**:
  * **Kafka Ingestion**: Acts as a message broker for high-velocity comment streams.
  * **Spark Analytics**: Stream-processes comments for real-time sentiment extraction.
* **Why it's advanced**: Engineered to scale for exceptionally high-traffic channels/live streams where typical single-process scripts would choke.
