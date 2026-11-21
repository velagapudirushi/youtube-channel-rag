"""
Built-in evergreen sample datasets for instant zero-wait demonstration and baseline coverage
on freeCodeCamp.org (@freecodecamp) and Andrej Karpathy (@AndrejKarpathy).
"""

from __future__ import annotations

SAMPLE_CHANNELS: dict[str, list[dict[str, str]]] = {
    "freeCodeCamp.org (@freecodecamp)": [
        {
            "video_id": "rfscVS0vtbw",
            "title": "Learn Python - Full Course for Beginners [Tutorial]",
            "channel": "freeCodeCamp.org",
            "url": "https://youtube.com/watch?v=rfscVS0vtbw",
            "transcript": (
                "Welcome to the freeCodeCamp complete Python programming course for beginners. "
                "In this course you will learn everything you need to get started with Python 3, from installing Python and PyCharm "
                "to variables, data types, strings, numbers, lists, tuples, functions, return statements, if statements, loops, "
                "dictionaries, try/except error handling, reading and writing external files, modules and pip, and object-oriented programming "
                "with classes, objects, and inheritance. Python is one of the most popular programming languages for web development, "
                "data science, machine learning, artificial intelligence, and automation. "
                "Lists in Python are mutable ordered sequences created with square brackets like my_list = [1, 2, 3], where you can modify, "
                "append, insert, remove, pop, sort, reverse, and slice elements in place. "
                "Tuples in Python are immutable ordered sequences created with parentheses like my_tuple = (1, 2, 3). "
                "The key difference between a Python list and a tuple is mutability: once a tuple is created, its elements cannot be changed, added, or removed. "
                "Because tuples are immutable, they are faster and more memory-efficient than lists, can be used as dictionary keys, and protect constant data such as coordinates from accidental modification. "
                "Functions allow you to organize reusable blocks of code using the def keyword, accept parameters, and return results."
            ),
        },
        {
            "video_id": "PkZNo7MFNFg",
            "title": "Learn JavaScript - Full Course for Beginners",
            "channel": "freeCodeCamp.org",
            "url": "https://youtube.com/watch?v=PkZNo7MFNFg",
            "transcript": (
                "This complete 134-part JavaScript tutorial from freeCodeCamp walks through everything from basic syntax to ES6 features. "
                "JavaScript is the scripting language of the web that runs directly in the browser and on the server via Node.js. "
                "In modern ES6 JavaScript, we declare variables using let for block-scoped mutable variables and const for constants, "
                "preferring them over var which is function-scoped and hoisted. "
                "We cover arrow functions, destructuring assignments, template literals, spread and rest operators, promises, async/await "
                "for asynchronous API calls, and DOM manipulation to build interactive frontend web applications."
            ),
        },
        {
            "video_id": "NWONeJKn6kc",
            "title": "Machine Learning for Everybody – Full Course",
            "channel": "freeCodeCamp.org",
            "url": "https://youtube.com/watch?v=NWONeJKn6kc",
            "transcript": (
                "Welcome to Machine Learning for Everybody on freeCodeCamp, taught by Kylie Ying. "
                "Machine learning is a subfield of artificial intelligence where computers learn patterns from data without being explicitly programmed. "
                "There are three main paradigms: Supervised Learning (where we train on labeled features X and target labels Y for classification and regression), "
                "Unsupervised Learning (where we discover hidden structure in unlabeled data using clustering like K-Means or dimensionality reduction like PCA), "
                "and Reinforcement Learning (where an agent learns to take actions in an environment to maximize cumulative reward). "
                "In supervised classification, we cover K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), "
                "and Neural Networks using TensorFlow and scikit-learn, evaluating models with precision, recall, F1-score, and confusion matrices."
            ),
        },
        {
            "video_id": "sVcwVQRHIc8",
            "title": "Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer",
            "channel": "freeCodeCamp.org",
            "url": "https://youtube.com/watch?v=sVcwVQRHIc8",
            "transcript": (
                "In this freeCodeCamp course taught by Lance Martin from LangChain, you will learn how to build Retrieval-Augmented Generation (RAG) systems from scratch. "
                "RAG connects Large Language Models to external knowledge sources like PDFs, documentation, databases, and YouTube transcripts. "
                "The core RAG pipeline has three stages: Indexing (loading documents, splitting them into chunks, embedding chunks into dense vectors, and storing them in a vector database), "
                "Retrieval (embedding the user's question and running k-nearest-neighbor cosine similarity search to fetch the most relevant chunks), "
                "and Generation (passing the retrieved chunks as context into the LLM prompt so the model generates a grounded, factual answer with source citations). "
                "Advanced RAG techniques covered in this course include Multi-Query retrieval, RAG-Fusion with Reciprocal Rank Fusion (RRF), "
                "HyDE (Hypothetical Document Embeddings), query routing, and parent-document retrieval."
            ),
        },
    ],
    "Andrej Karpathy (@AndrejKarpathy)": [
        {
            "video_id": "kCc8FmEb1nY",
            "title": "Let's build GPT: from scratch, in code, spelled out",
            "channel": "Andrej Karpathy",
            "url": "https://youtube.com/watch?v=kCc8FmEb1nY",
            "transcript": (
                "Hello everyone, welcome! Today we are going to build GPT from scratch, in code, spelled out. "
                "We will follow the seminal paper Attention Is All You Need and OpenAI's GPT-2 and GPT-3 papers. "
                "At the heart of the Transformer architecture is the self-attention mechanism. Why do we need attention mechanisms? "
                "In a language model, tokens in a sequence need to communicate with each other to aggregate information from the past. "
                "Self-attention is a communication mechanism where every token emits three vectors: a Query vector, a Key vector, and a Value vector. "
                "The Query vector says what am I looking for? The Key vector says what do I contain? "
                "We take the dot product between the Queries and the Keys, divide by the square root of the head dimension d_k for scaled dot-product attention, "
                "apply a causal mask so future tokens cannot communicate with past tokens in autoregressive decoding, and then softmax to get attention weights. "
                "Finally, we aggregate the Value vectors using those attention weights. "
                "Multi-head attention simply runs multiple self-attention heads in parallel and concatenates their outputs, allowing the Transformer "
                "to attend to information from different representation subspaces at different positions. "
                "After communication via multi-head self-attention, each token processes what it gathered independently through a feed-forward multilayer perceptron (MLP). "
                "Crucially, residual connections (skip highways) and LayerNorm (pre-norm formulation) make deep Transformers trainable."
            ),
        },
        {
            "video_id": "zjkBMFhNj_g",
            "title": "Intro to Large Language Models",
            "channel": "Andrej Karpathy",
            "url": "https://youtube.com/watch?v=zjkBMFhNj_g",
            "transcript": (
                "In this talk we give a general introduction to Large Language Models (LLMs). "
                "An LLM really consists of two files: a parameters file containing billions of float16 or int8 weights, and a small code file that runs the neural network architecture, almost always a Transformer. "
                "Training an LLM happens in stages. Stage 1 is Pre-training: compressing terabytes of internet text into neural network weights using next-token prediction. "
                "This costs millions of dollars and thousands of GPUs, producing a base model that is essentially a document completer. "
                "Stage 2 is Fine-tuning and Alignment (Supervised Fine-Tuning or SFT, followed by RLHF — Reinforcement Learning from Human Feedback). "
                "This turns the base model into a helpful assistant that answers questions and follows instructions. "
                "We can also think of LLMs as the kernel process of an emerging Operating System: the context window acts as RAM (working memory), "
                "Retrieval-Augmented Generation (RAG) acts as the file system or disk storage where we fetch relevant documents into the context window, "
                "and tools like code interpreters, calculators, and web search act as system calls."
            ),
        },
        {
            "video_id": "7xTGNNLPyMI",
            "title": "Deep Dive into LLMs like ChatGPT",
            "channel": "Andrej Karpathy",
            "url": "https://youtube.com/watch?v=7xTGNNLPyMI",
            "transcript": (
                "Welcome to a comprehensive deep dive into Large Language Models like ChatGPT. "
                "When you ask a question about proprietary data or specific videos, you should use Retrieval-Augmented Generation (RAG). "
                "Why does RAG work so well? Because anything placed directly inside the Transformer's context window is immediately accessible to the self-attention heads as working memory, "
                "whereas information vaguely stored in the model parameters from pre-training can suffer from hallucinations. "
                "By chunking documents or transcripts, embedding them into a vector database, retrieving the top matching chunks, and stuffing them into the prompt context, "
                "the LLM can generate grounded, verifiable answers with citations. "
                "We also discuss reasoning models, test-time compute, reinforcement learning, and how models learn to think step-by-step using chain-of-thought tokens before emitting a final response."
            ),
        },
        {
            "video_id": "zduSFxRajkE",
            "title": "Let's build the GPT Tokenizer",
            "channel": "Andrej Karpathy",
            "url": "https://youtube.com/watch?v=zduSFxRajkE",
            "transcript": (
                "Today we are going to build the GPT Tokenizer from scratch using Byte Pair Encoding (BPE). "
                "Tokenization is the translation layer between raw UTF-8 strings and the discrete integer token IDs that Transformers actually see. "
                "Many quirky LLM failure modes—like struggling to spell words backward, counting letters in 'strawberry', doing arithmetic, or handling non-English languages and YAML whitespace—actually trace right back to tokenization! "
                "In Byte Pair Encoding, we start with raw bytes (256 tokens) and iteratively merge the most frequent adjacent pair of tokens into a new vocabulary token until we reach our target vocabulary size, such as 50,257 for GPT-2 or 100,277 for GPT-4 cl100k_base."
            ),
        },
    ],
}

SAMPLE_KARPATHY_VIDEOS = SAMPLE_CHANNELS["Andrej Karpathy (@AndrejKarpathy)"]
