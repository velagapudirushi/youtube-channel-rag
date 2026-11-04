# RAG Evaluation Report: `youtube-channel-rag`

- **Dataset:** `youtube-channel-rag-benchmark-v1` (7 queries, cutoff `K=5`)
- **Overall Pass Rate:** **0.0%** (0/7)
- **Cross-Checked Faithfulness:** **52.8%** (LLM-Judge: `57.7%` vs. NLI-Entailment: `54.1%`)
- **Judge-NLI Disagreements Flagged:** `1` / `69` atomic claims (`1.5%`)
- **Mean Latency:** `2.9 ms`

## 1. Retrieval & Faithfulness Summary

| Category | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **Retrieval** | `Recall@5` | **0.0%** | Fraction of golden chunks retrieved in top-5 |
| **Retrieval** | `Precision@5` | `0.0%` | Signal-to-noise ratio in top-5 window |
| **Retrieval** | `MRR` | **0.000** | Reciprocal rank of first golden chunk |
| **Retrieval** | `NDCG@5` | **0.000** | Rank-aware retrieval quality |
| **Retrieval** | `Context Fact Coverage` | `100.0%` | Reference key facts present in retrieved text |
| **Faithfulness** | `Method A: LLM-as-a-Judge` | `57.7%` | Holistic passage-level judge |
| **Faithfulness** | `Method B: NLI Entailment` | `54.1%` | Sentence-aligned premise-hypothesis cross-encoder |
| **Faithfulness** | **`Cross-Checked Score`** | **52.8%** | Calibrated ensemble penalizing NLI contradictions |

## 2. Representative Failure Taxonomy

### `RETRIEVAL_MISS` — Retrieval Miss (Golden Evidence Absent from Top-K Chunks) (6 queries / 85.7%)

> **Why this happens:** The retriever failed to surface the ground-truth document chunk in the top-K results, starving the generator of required facts.
>
> **Recommended Fix:** Switch from pure dense embeddings to Hybrid Search (BM25 + Dense with Reciprocal Rank Fusion) and add query expansion for exact error codes, CLI flags, and acronyms.

#### Representative Failure #1: `fcc_01_python_list_tuple` — *What does this channel teach about Python lists vs tuples and mutability?*
- **Retrieved Chunks:** `['chunk_1', 'chunk_2']` (Golden: `['rfscVS0vtbw_chunk_0', 'rfscVS0vtbw_chunk_1']`, First Hit Rank: `MISSED`)
- **Generated Answer:** ### Grounded Answer from **freeCodeCamp.org**

- The key difference between a Python list and a tuple is mutability: once a tuple is created, its elements cannot be changed, added, or removed. — *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*
- In this course you will learn everything you need to get started with Python 3, from installing Python and PyCharm to variables, data types, strings, numbers, lists, tuples, functions, return statements, if statements, loops, dictionaries, try/except error handling, reading and writing external files, modules and pip, and object-oriented programming with classes, objects, and inheritance. — *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*
- Tuples in Python are immutable ordered sequences created with parentheses like my_tuple = (1, 2, 3). — *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*
- Because tuples are immutable, they are faster and more memory-efficient than lists, can be used as dictionary keys, and protect constant data such as coordinates from accidental modification. — *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*
- Lists in Python are mutable ordered sequences created with square brackets like my_list = [1, 2, 3], where you can modify, append, insert, remove, pop, sort, reverse, and slice elements in place. — *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*
- **Ground Truth:** Lists in Python are mutable ordered sequences created with square brackets where elements can be modified, appended, or removed, whereas tuples are immutable ordered sequences created with parentheses whose elements cannot be changed once created, making tuples faster and usable as dictionary keys.
- **Faithfulness Breakdown:** LLM Judge `0.45` | NLI Entailment `0.50` | **Cross-Checked `0.44`**
- **Root Cause Diagnosis:** Retriever missed golden chunk(s) ['rfscVS0vtbw_chunk_0', 'rfscVS0vtbw_chunk_1'] (Recall@5=0.00), causing generator to hallucinate unsupported facts (Faithfulness=0.44).
- **Flagged Atomic Claims:**
  - `"### Grounded Answer from **freeCodeCamp.org**."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.16`, `C=0.05`)
  - `"— *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.21`, `C=0.12`)
  - `"— *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.21`, `C=0.12`)
  - `"— *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.21`, `C=0.12`)
  - `"— *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.21`, `C=0.12`)
  - `"Lists in Python are mutable ordered sequences created with square brackets like my_list = [1, 2, 3], where you can modify, append, insert, remove, pop, sort, reverse, and slice elements in place."` → LLM: `CONTRADICTED`, NLI: `ENTAILMENT` (`E=0.98`, `C=0.01`) ⚠️ **[LLM_FALSE_NEGATIVE]**
  - `"— *[Learn Python - Full Course for Beginners [Tutorial]](https://youtube.com/watch?v=rfscVS0vtbw)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.21`, `C=0.12`)

#### Representative Failure #2: `fcc_02_javascript_es6` — *How are variables declared in modern ES6 JavaScript and what asynchronous features are covered?*
- **Retrieved Chunks:** `['chunk_1']` (Golden: `['PkZNo7MFNFg_chunk_0']`, First Hit Rank: `MISSED`)
- **Generated Answer:** ### Grounded Answer from **freeCodeCamp.org**

- In modern ES6 JavaScript, we declare variables using let for block-scoped mutable variables and const for constants, preferring them over var which is function-scoped and hoisted. — *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*
- This complete 134-part JavaScript tutorial from freeCodeCamp walks through everything from basic syntax to ES6 features. — *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*
- JavaScript is the scripting language of the web that runs directly in the browser and on the server via Node.js. — *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*
- We cover arrow functions, destructuring assignments, template literals, spread and rest operators, promises, async/await for asynchronous API calls, and DOM manipulation to build interactive frontend web applications. — *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*
- **Ground Truth:** In modern ES6 JavaScript, variables are declared using let for block-scoped mutable variables and const for constants instead of var, and asynchronous API calls use promises and async/await.
- **Faithfulness Breakdown:** LLM Judge `0.53` | NLI Entailment `0.46` | **Cross-Checked `0.45`**
- **Root Cause Diagnosis:** Retriever missed golden chunk(s) ['PkZNo7MFNFg_chunk_0'] (Recall@5=0.00), causing generator to hallucinate unsupported facts (Faithfulness=0.45).
- **Flagged Atomic Claims:**
  - `"### Grounded Answer from **freeCodeCamp.org**."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.16`, `C=0.05`)
  - `"— *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)
  - `"— *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)
  - `"— *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)
  - `"— *[Learn JavaScript - Full Course for Beginners](https://youtube.com/watch?v=PkZNo7MFNFg)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)

#### Representative Failure #3: `karpathy_03_bpe_tokenizer` — *How does Byte Pair Encoding (BPE) tokenization work in GPT models?*
- **Retrieved Chunks:** `['chunk_1']` (Golden: `['zduSFxRajkE_chunk_0']`, First Hit Rank: `MISSED`)
- **Generated Answer:** ### Grounded Answer from **Andrej Karpathy**

- Today we are going to build the GPT Tokenizer from scratch using Byte Pair Encoding (BPE). — *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*
- In Byte Pair Encoding, we start with raw bytes (256 tokens) and iteratively merge the most frequent adjacent pair of tokens into a new vocabulary token until we reach our target vocabulary size, such as 50,257 for GPT-2 or 100,277 for GPT-4 cl100k_base. — *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*
- Tokenization is the translation layer between raw UTF-8 strings and the discrete integer token IDs that Transformers actually see. — *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*
- Many quirky LLM failure modes—like struggling to spell words backward, counting letters in 'strawberry', doing arithmetic, or handling non-English languages and YAML whitespace—actually trace right back to tokenization! — *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*
- **Ground Truth:** Byte Pair Encoding starts with 256 raw byte tokens and iteratively merges the most frequent adjacent pair of tokens into new vocabulary tokens until reaching target vocabulary size such as 50,257 for GPT-2 or 100,277 for GPT-4.
- **Faithfulness Breakdown:** LLM Judge `0.53` | NLI Entailment `0.48` | **Cross-Checked `0.46`**
- **Root Cause Diagnosis:** Retriever missed golden chunk(s) ['zduSFxRajkE_chunk_0'] (Recall@5=0.00), causing generator to hallucinate unsupported facts (Faithfulness=0.46).
- **Flagged Atomic Claims:**
  - `"### Grounded Answer from **Andrej Karpathy**."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)
  - `"— *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.18`, `C=0.05`)
  - `"— *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.18`, `C=0.05`)
  - `"— *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.18`, `C=0.05`)
  - `"— *[Let's build the GPT Tokenizer](https://youtube.com/watch?v=zduSFxRajkE)*."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.18`, `C=0.05`)

### `INTRINSIC_CONTRADICTION` — Intrinsic Contradiction (Misread Numbers, Negations, or Tiers) (1 queries / 14.3%)

> **Why this happens:** The retriever fetched the relevant passage, but the generator misread a numeric figure, inverted a negation polarity, or confused policy rules across adjacent tiers.
>
> **Recommended Fix:** Keep structured tables/policies intact during chunking (semantic/markdown-header chunking) and instruct the generator to verify entity scope before stating numbers.

#### Representative Failure #1: `karpathy_01_self_attention` — *How does self-attention work in a Transformer using Query, Key, and Value vectors?*
- **Retrieved Chunks:** `['chunk_1', 'chunk_2']` (Golden: `['kCc8FmEb1nY_chunk_0', 'kCc8FmEb1nY_chunk_1']`, First Hit Rank: `MISSED`)
- **Generated Answer:** ### Grounded Answer from **Andrej Karpathy**

- Self-attention is a communication mechanism where every token emits three vectors: a Query vector, a Key vector, and a Value vector. — *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*
- At the heart of the Transformer architecture is the self-attention mechanism. — *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*
- We take the dot product between the Queries and the Keys, divide by the square root of the head dimension d_k for scaled dot-product attention, apply a causal mask so future tokens cannot communicate with past tokens in autoregressive decoding, and then softmax to get attention weights. — *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*
- Finally, we aggregate the Value vectors using those attention weights. — *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*
- Multi-head attention simply runs multiple self-attention heads in parallel and concatenates their outputs, allowing the Transformer to attend to information from different representation subspaces at different positions. — *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*
- **Ground Truth:** Every token emits Query, Key, and Value vectors. We compute scaled dot-product attention between Queries and Keys, apply a causal mask and softmax to get attention weights, and aggregate the Value vectors.
- **Faithfulness Breakdown:** LLM Judge `0.47` | NLI Entailment `0.45` | **Cross-Checked `0.45`**
- **Root Cause Diagnosis:** Generated answer contradicts retrieved evidence (5 contradicted claim(s)): "— *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*."
- **Flagged Atomic Claims:**
  - `"### Grounded Answer from **Andrej Karpathy**."` → LLM: `UNVERIFIABLE`, NLI: `NEUTRAL` (`E=0.06`, `C=0.06`)
  - `"— *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*."` → LLM: `CONTRADICTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.91`)
  - `"— *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*."` → LLM: `CONTRADICTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.91`)
  - `"— *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*."` → LLM: `CONTRADICTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.91`)
  - `"— *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*."` → LLM: `CONTRADICTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.91`)
  - `"— *[Let's build GPT: from scratch, in code, spelled out](https://youtube.com/watch?v=kCc8FmEb1nY)*."` → LLM: `CONTRADICTED`, NLI: `CONTRADICTION` (`E=0.03`, `C=0.91`)
