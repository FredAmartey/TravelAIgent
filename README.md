# TravelAIgent - AI Powered Travel Guide (RAG Pipeline with Fine-Tuned LLaMa 2)
Travel planning can be overwhelming, with users spending hours researching destinations, activities, and accommodations across multiple platforms. The AI-Powered Travel Guide aims to streamline this process by combining Retrieval-Augmented Generation (RAG), Fine-tuning, and Prompt Engineering to deliver highly personalized travel recommendations and itineraries. The system will cater to user preferences such as budget, travel style, and specific interests (e.g., food, adventure, culture).

Video demo: https://theamarteyboy2.wistia.com/medias/00r620t58t


## **Project Overview**
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** using **LangChain** and a **fine-tuned LLaMa 2 model** to answer user queries related to USA or Canadian travel policies and guidelines.  
The pipeline leverages **Pinecone** as a vector database and a classification LLM to identify the relevant context (USA or Canada).  

The LLaMa 2 model has been fine-tuned on a specific dataset hosted on HuggingFace to enhance its accuracy for travel-related questions.

---

## **File Structure**

1. **`Finetuning_Llama_2_using_QLORA.ipynb`**  
   - Contains the code for fine-tuning the LLaMa 2 model using QLoRA (Parameter Efficient Fine-Tuning) on **Google Colab Free Tier**.  
   - The model is trained on a HuggingFace dataset, optimized for travel policy question answering.  

2. **`ingest.py`**  
   - Handles document ingestion into the **Pinecone** vector database.  
   - Steps include:  
     - Extracting text from relevant PDF documents.  
     - Splitting text into smaller chunks for embedding.  
     - Generating embeddings using a HuggingFace embedding model.  
     - Uploading embeddings to the Pinecone database under appropriate namespaces (`usa_docs` and `canada_docs`).

3. **`RAG_on_fine_tuned_LLAMA_model.ipynb`**  
   - Implements the RAG pipeline to answer user queries.  
   - Steps include:  
     1. **Model Initialization**:  
        - Fine-tuned **LLaMa 2** for answer generation.  
        - **Mixtral** model (Groq) to classify queries as "USA" or "Canada".  
        - HuggingFace embedding model for embedding user queries.  
     2. **Classification**:  
        - The query is classified to determine whether it pertains to **USA** or **Canada** travel guidelines.  
     3. **Retrieval**:  
        - Based on the classification result, the pipeline connects to the relevant Pinecone **namespace** (`usa_docs` or `canada_docs`).  
        - Retrieves the most relevant document chunks using a vectorstore-backed retriever.  
     4. **Answer Generation**:  
        - A LangChain prompt template combines the retrieved context and the user query.  
        - The fine-tuned LLaMa 2 generates the final response.  

4. **`requirements.txt`**  
   - Contains all the dependencies required to run the project, including:  
     - LangChain  
     - Pinecone  
     - HuggingFace Transformers  
     - Groq (for Mixtral model)  
     - PyPDFLoader (for PDF processing)  

---

## Setup instructions
1. You can not run it locally if you dont have a gpu. The two colab files, both use llm model from huggingface that requires gpu, so you can run them via the google colab link.
2. The ingest.py file is for uploading the pdf documents to pinecone database. I already ran that file and ingested the information, so you shouldn't run it again, as that would introduce duplicates to the pinecone database.
3. The ONLY file you should run is the RAG colab file. It just connects to pinecone, loads llm using gpu, and answers your travel related questions. As for the fine-tuning file, you shouldnt run it as that would re-start the fine-tuning process


## **Key Features**
1. **Fine-Tuned LLaMa 2**  
   - LLaMa 2 is fine-tuned specifically for travel-related queries, ensuring high-quality answers.  
   - QLoRA fine-tuning method optimizes training for limited resources.

2. **Query Classification**  
   - Utilizes Mixtral model (Groq) to classify queries as "USA", "Canada", "Europe", "Asia", etc.  
   - Enables accurate retrieval of relevant context.  

3. **Efficient Retrieval**  
   - PDF documents are ingested into Pinecone with namespace mapping for **USA**, **Canada**, **Europe**, **Asia**, etc guidelines.  
   - Embedding-based retrieval ensures the most relevant document chunks are fetched.  

4. **RAG Pipeline**  
   - Combines retrieved context and fine-tuned LLaMa 2 to generate precise, context-aware answers.  

5. **Scalability**  
   - Uses Pinecone for scalable vector storage and retrieval.  
   - Modular pipeline that can be extended for additional namespaces or models.  

---

## **How It Works**

### **1. Document Ingestion**  
- Run the `ingest.py` file to process relevant PDF documents and upload embeddings to Pinecone.  
- Document chunks are stored under separate namespaces:  
   - `usa_guides` for USA travel guidance.  
   - `canada_guides` for Canadian travel guidance.
   - `european_guides` for Europe travel guidance. 

### **2. RAG Pipeline**  
- Run the `RAG_on_fine_tuned_LLAMA_model.ipynb` notebook.  
- User query is:  
   - **Classified**: Determines if it's related to USA or Canada.  
   - **Retrieved**: Connects to Pinecone to fetch context from the relevant namespace.  
   - **Answered**: Combines retrieved context with the query, and the fine-tuned LLaMa 2 model generates the answer.

---

## Dataset information
The dataset was taken from Huggingface having around 1000 samples. The link is https://huggingface.co/mlabonne/llama-2-7b-guanaco

## Acknowledgments
HuggingFace for pre-trained models and datasets.
Pinecone for scalable vector storage.
Groq for providing Mixtral LLM.

## Target Users
- Frequent travelers seeking quick and tailored itineraries.
- Travel agencies looking for AI-powered tools to enhance customer engagement.
- First-time travelers needing curated suggestions.

## Expected Impact
- Reduce time spent on travel planning.
- Provide hyper-personalized, dynamic itineraries based on user input.
- Enhance user satisfaction by offering reliable and comprehensive travel insights.

## Implementation Strategy

### Retrieval-Augmented Generation (RAG)
#### Vector Database
- Use Pinecone or Weaviate to store structured and unstructured travel data, including:
  - Destination guides (e.g., famous landmarks).
  - Restaurant reviews (e.g., Yelp or Google Places).
  - Activities and cultural events.

#### Retrieval System
- Implement semantic search to fetch contextually relevant data.
  - Example: Query "best activities in Paris for food lovers" retrieves data on food tours, cooking classes, and local restaurants.

#### Integration with LLM
- Combine retrieved data with a pre-trained LLM (e.g., OpenAI GPT-4 or a fine-tuned version) to generate cohesive, natural responses.

### Fine-Tuning Approach
#### Dataset Selection
- Use publicly available datasets (e.g., Wikivoyage, TripAdvisor reviews, curated travel blogs).
- Collect domain-specific data, including regional travel preferences and itineraries.

#### Fine-Tuning Process
- Fine-tune an LLM (e.g., using Hugging Face Transformers) to adapt it for travel-specific conversational queries.
- Focus on improving accuracy and diversity of responses.

### Prompt Engineering Strategies
#### Dynamic Prompt Templates
- Use templates for structured responses, such as:
  - "Plan a [duration]-day trip to [destination] for a [travel style] traveler. Include activities, accommodations, and dining options."
  - "Suggest off-the-beaten-path activities for [destination]."

#### Chain-of-Thought Prompting
- Use step-by-step reasoning to improve response coherence for complex queries like multi-city trips.

## Technical Overview


#### Backend
- **LLM**: OpenAI GPT-4 or a fine-tuned model for natural language understanding and response generation.
- **Vector Database**: Pinecone for semantic retrieval of travel data.

### Tools and Frameworks
- **LLM**: OpenAI, Hugging Face Transformers.
- **Vector Database**: Pinecone, Weaviate, or Milvus.
- **RAG Framework**: LangChain for LLM and retrieval integration.

### Data Sources and Preparation
- Public travel datasets (e.g., Wikivoyage, TripAdvisor, OpenStreetMap).
- Preprocessing: Clean and tokenize textual data, removing duplicates or irrelevant entries.

### Integration Strategy
1. Implement RAG for fetching relevant data based on user queries.
2. Fine-tune the LLM to enhance travel-specific responses.
3. Use prompt templates to structure interactions and maintain user context.
4. Seamlessly combine components into a user-friendly application.

## Project Milestones

| Milestone                             | Timeline       | Description                                             |
|---------------------------------------|----------------|---------------------------------------------------------|
| **RAG Implementation**               | Weeks 1-2      | Set up a vector database and build a retrieval pipeline.|
| **Fine-Tuning Process**               | Weeks 3-4      | Fine-tune the LLM using curated travel datasets.        |
| **Prompt Engineering Development**    | Week 5         | Design and test advanced prompt templates.             |
| **System Integration and Testing**    | Week 6         | Integrate RAG, fine-tuned LLM, and prompt engineering.  |

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
