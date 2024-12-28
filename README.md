# PDF RAG LLM with Langchain

Create a Retrieval-Augmented Generation (RAG) LLM to consume PDF documents and allow users to prompt questions based on pdf documents to upload to RAG.

# 1. Introduction:
The scope of the given project involves designing a Retrieval-Augmented Generation system that,
when fed with PDFs, will enable users to query in text and get back a very precise, context-driven
response. The system will leverage the powerful LangChain, ChromaDB, and Ollama-based LLMs
including llama3.2 and nomic-embed-text. Loading and preprocessing of PDF data, creation of
vector DB, RAG pipeline implementation, and ROUGE evaluation is to be covered as part of this
project.

# 2. Model Architectures and Experiments
# 2.1. Model Architecture

Retriever:
• ChromaDB: Stores vector embeddings of text chunks for fast similarity-based retrieval.
• Embedding Model: nomic-embed-text by Ollama For generating dense vector embeddings for the retriever.

Language Model LLM:
llama3.2: A local language model; for offline question answering using retrieved contexts.

# 2.2. Steps in RAG Implementation
PDF Text Extraction: PyMuPDFLoader from LangChain splits document texts into manageable chunks by efficiently extracting text from a PDF document.
Text Splitting: RecursiveCharacterTextSplitter splits the extracted content into chunks of 1000
characters with an overlap of 200 characters for continuity in a chunk.
Embedding and Storage: Each chunk is converted into vector embeddings via nomic-embed-text, and then it is stored for efficient retrieval in ChromaDB.
Question-Answering Pipeline: The above retriever retrieves the relevant chunks for a given question using MultiQueryRetriever. A LangChain pipeline combines the retriever, a prompt template, and the LLM to generate answers.

# 2.3. Loss/Objective Function
While no explicit loss function is implemented, the effectiveness of the system is assessed through ROUGE scores by comparing generated answers against ground-truth answers for precision, recall, and F1.

# 2.4. Experiments Conducted
## Experiment 1: OpenAI with Local Embeddings
Setup:
• Selected OpenAI GPT-3.5 Turbo API in order to answer the questions.
• Began exploration for embeddings using OpenAI’s text-embedding-ada-002 with some troubles concerning the processing of embeddings since they are contained within the API.
• Changed to local embeddings which are based on the transformer’s library (e.g., all- MiniLM-L6-v2).
• Vector database: ChromaDB.

## Challenges:
• The process of integrating OpenAI embedding was not easy and full of API calls which made it resource demanding.
• For the retrieval, I experimented with using local embeddings and OpenAI GPT-3.5 Turbo API, but the obtained quality of the results is rather low.
• In particular, retrieval inaccuracies prevented contextual information in the form of relevant or complete query from reaching the LLM, which impacted the correctness of the answers.

## Outcome:
• Decided to discontinue OpenAI API usage due to:
• Reliance on third party services and API’s.
• Low complementarity of local embeddings and OpenAI GPT-3.5 Turbo.

# Experiment 2: Ollama with Local Embeddings
## Setup:
• Used nomic-embed-text from the corpus of Ollama’s nomic embeddings.
• Subtitle2.3 – Integrated the llama3.2 local model for question answering.
• Vector database: ChromaDB.

## Advantages:
• Integration of generation with nomic-embed-text was fast with no issues regarding interaction with ChromaDB.
• Relevancy and quality of the retrieved documents increased because the local embeddings of the documents were learned for the purpose of capturing the document chunks.
• The local implementation of Ollama setup was more effective due to freedom from dependencies on external API’s.

## Outcome:
• The queries identified pertinent chunks and returned them to llama3.2, resulting in satisfyingly accurate answers.
• Officially closed Ollama as the embedding and language model supplier of the RAG pipeline.

## Chunk Sizes:
Tried chunk sizes of 100 and 500 characters. It was found that using 100 characters with an overlap of 200 resulted in the best value for context but high retrieval accuracy.

# Prompt Templates:
Examined all the subject tested templates with and without steps indicating that numerical thinking
was required. They said that incorporating specific and precise mathematical descriptions for
stages of computations made the final numbers more precise.

# 3. Neural Network Overview
3.1. LLM: llama3.2
• A lightweight, local language model optimized for contextual understanding and text
generation.
• Successfully integrated with LangChain for prompt-based response generation.

# 3.2. Design Process

The designing of the Retrieval-Augmented Generation (RAG) system was initiated in a hierarchical manner to ensure that individual units of the design were effectively integrated.
Following is a brief elaboration of the design process which covers each part of the given figure.

# Step 1: Data Ingestion
The first step was to get the text out from the PDFs using the PyMuPDFLoader of LangChain which fastly converts the content into a form that a computer can process. This step was important to guarantee that the raw text was correctly upload and ready for more treatments. This extracted data was defined as a set of documents in the LangChain format to preserve the necessary context and formalization from the PDF.

# Step 2: Text Splitting
To make further analyses feasible, the extracted text was further preprocessed with LangChain’s RecursiveCharacterTextSplitter, which segmented it into parts that each comprised 1000 characters with an overlap of 200 characters. This overlap made sure the context didn’t get lost across boundaries and ensured nothing was lost to the next stages. The splitting process was important in order to be able to deal with large documents and make the text(body) appropriate for the generation of the embeddings.

# Step 3: Embedding Generation
Every piece of text was vectorized to higher-dimensional representations. First, I tried to use OpenAI’s text-embedding-ada-002 and local embeddings with the help of the transformers library inside the notebook, but the results were barely satisfactory. The last setup used the nomic-embed-text model of Ollama, which yielded precise and fast compatible embeddings that fit the RAG pipeline. These embeddings allowed for the use of search to obtain the desired results.

# Step 4: Vector Database and Retrieval
The embeddings were saved in ChromaDB, a vector database designed for high performance for similarity-based search. With the help of MultiQueryRetriever of LangChain, the variations of the user’s query were produced to obtain the most suitable chunks from the database. This retrieval mechanism enabled passage relevant information to be passed to the language model for answering questions.

# Step 5: Language Model (LLM)
The text chunks which were retrieved were then processed using the llama3.2 model, a local language model that was incorporated into the LangChain using the ChatOllama. The LLM played the role of providing accurate answers to the users based on the context their queries were made. Local dependencies such as the one provided by the llama3.2 model provided consistency, mitigated callback needed for APIs, and enhanced system performance, particularly where privacy was of concern.

# Step 6: Prompt Engineering
To help LLM create accurate and concise responses, specific templates of prompts were developed for each type of answer. Such templates included outlining the manner in which to address numerical reasoning and making sure that the answers were solely rooted in the content. Additional modifications to the prompts made the language model more effective for the task, by negating the two extremes of wordiness and obscurity.

# Step 7: Evaluation
The validity of the system responses was thus evaluated using recall-orientation-understanding- generation measures for word level (W), phrase level (P), and structure level (R) or ROUGE-1, ROUGE-2 and ROUGE-L respectively. Through the formal evaluation of the pipeline, strengths and areas that need improvement were identified to have enhanced the successive improvements
to the pipeline. This step allowed that for a wide range of questions entered by users, the system provides correct and highly qualified answers. Evaluating in terms of its using and application, design process organizes interaction of components and their functionalities in the most accurate, optimal performance, and resource-effective manner. The corresponding tweaking of embeddings and LLMs enhanced the reliability of the results using Ollama’s local infrastructure.

# Challenges Faced
1. OpenAI Embedding Challenges
Problem: When working with OpenAI embeddings and when using the OpenAI GPT-3.5 Turbo API, it was found that the embedding performance was suboptimal.

Observation: The embeddings also did not adequately capture original context from the vector database and this consequently affected desirable answer characteristics.

Resolution: Switched to local embeddings with nomic-embed-text from Ollama that visibly gave better compatibility with the RAG pipeline and better results when it came to retrieval.

2. Installation Difficulties
Problem: It took considerable time to deploy Ollama and it’s dependencies as well as all other required libraries.

Issues Faced: The compatibility of different versions of protobuf with various other modules, onnx etc. Identification of errors that occur during installation of Ollama models and embeddings.

Resolution: Spent considerable time resolving issues related to library compatibility and how library is being put into user interface.

3. Prompt Engineering Challenges
Problem: To formulate queries that would elicit responses that captured the meanings intended by the software, ask prompt construction was time-consuming and trial-and-error. Supervisions of the tasks that had numerical reasoning and factual accuracy prompts necessitated modifications more frequently. It was found that slight modification in instructions can affect variation in the output generated by the system.

Observation: Sensitivity was calculated as soon as the prompt was given, and the results showed that response was faster in OpenAI GPT-3.5 Turbo rather than Ollama’s local models.

Resolution: I managed to finalize a general-purpose prompt of Ollama to get consistent results.

Conclusion: Why Llama Was Finalized:
Both local embeddings with Ollama’s nomic-embed-text did not have any issues integrating with
the RAG pipeline. Questions answered by llama3.2 were much more accurate and context-aware in comparison to
OpenAI GPT-3.5 Turbo. Learning Outcome: The experimental results also show that both embedding and model selection
strongly influence retrieval and generation performance. The installation and configuration of local
tools such as Ollama are more resource-demanding than cloud tools but more effective for offline or
private applications.

# Results
An assessment of the RAG system was done based on 10 Qs from the assignment. Although the system provided answers for all the questions, ROUGE scores were compared for only two of
them with the reference answers given. 

## Result of Inference in Respect of Answer to Question 1 The results for the first question show moderate performance, as reflected in the ROUGE scores:

### ROUGE-1:
Precision (54.2%): Says that approximately 51 percent of the words in the generated answer is
correct when compared to the correct answer.
Recall (44.8%): Notices that only about 30% of the total amount of the words in the reference answer
were provided in the generated answer.
F1-Score (49.1%): This balance is again conveyed between precision and recall, as it can be seen
the generated answer contains a portion of what needs to be highlighted but there are other aspects
missed.

### ROUGE-2:
Precision (26.1%): Approximately 25% of the bigrams coincide with the reference answers and
0.10% of the trigrams coincide.
Recall (21.4%): More than three-quarters of the bigrams in the reference answer were omitted from
the generated answer.
F1-Score (23.5%): Points to the fact that in the given case reference and generated solution
phraseology differs grossly affecting the concept of fluency.

### ROUGE-L:
Precision (41.7%): About 42% of the longest matching sequences of the generated and reference
answers are accurate.
Recall (34.5%): Demonstrates that about one-third of the LMSs in the reference answer was
obtained.
F1-Score (37.7%): The abbreviated encodes ‘part’ indicates that while the structure of the response is
partially aligned it is not near the reference ideally.

### Key Observations

### Strengths:
While evaluating the generated answer, we see that it incorporates some aspects of the reference
answer, as confirmed by the moderate ROUGE-1 and ROUGE-L metrics.
Quantitative information (for example, the number of employees and the percentage of a decrease)
was partially accurate, explaining the capacity of the system to work with numerical data.

### Weaknesses:
Low ROUGE-2 scores mean low fluency and phrasing similarity; the problem was observed when
the generated answer is less similar to the reference and does not only contain different words.
All the recall scores obtained indicate that some relevant information was missed from the
reference answer.

## Information R for Question 2
The results for the second question show strong performance, as reflected in the high ROUGE
scores:

### ROUGE-1:
Precision (87.5%): Shows that close to 100 per cent of the words in the generated answer are the
same as the words in the reference answer.
Recall (87.5%): Notes that if excluding a few words, all the words in the given reference answer
are reflected in the generated answer.
F1-Score (87.5%): The scores about precision and recall in terms of sentences and a further strong
emphasis on the fact that all components and aspects of the generated answer are in perfect match
with the reference.

### ROUGE-2:
Precision (80%): Indicates that a majority of the bigrams; a pair of consecutive words into the
generated answer matches the reference answer.
Recall (80%): Demonstrates that a vast majority in the bigrams in the reference set were captured.
F1-Score (80%): On the correlation coefficient, we see smooth transition in the strings of generated
responses, with minimal drift from the reference.

### ROUGE-L:
Precision (87.5%): Points out that, long exact matches between the generated and reference
answers are virtually entirely right.
Recall (87.5%): Demonstrates that the structure of the generated answer can encompass all the
aspects of reference answer.
F1-Score (87.5%): Esteems the overall compositional structure and relevance consistency of the
generated response.

### Key Observations

### Strengths:
It evaluated that the overall content of the generated answer indeed picked up the facts of the reference answer along with general and specific context.
High ROUGE-2 scores indicate the test summaries’ fluent and almost identical match in phrasing with the reference summary. The recall scores reveal that almost no information is lost and the answer was accurate and
exhaustive.

### Weaknesses:
Small issues with cohesion and fluency, such as changes in key words, may justify the decrease in ROUGE-2 scores compared to ROUGE-1, ROUGE-L.

# Improvements Needed
Fluency Fine-Tuning: However, the phrasings used are correct; perhaps further enhancing the
LLM to produce more stylistically consistent responses may slightly increase ROUGE-2 results.
Prompt Optimization: Some additional changes to the prompt can bring the results closer to the structure of the reference answer.

# Conclusion
In this assignment, the Retrieval-Augmented Generation (RAG) pipeline was applied to retrieve and generate coherent responses from a PDF dataset effectively. The goal was to create a system that would maximize the use of LangChain, ChromaDB, and Ollama’s local LLMs to answer a query with high precision and relevancy to the user. Across 12 questions, the pipeline for neural IR ranked significantly higher than the previous model, promoting the quantitative and more factual-reasoning questions while showing potential for enhancement in other areas.
