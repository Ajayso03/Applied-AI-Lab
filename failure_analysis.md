# DisasterRAG – Retrieval Failure Analysis

This document provides a detailed, empirical forensic analysis of two genuine retrieval failure cases identified during the evaluation of **DisasterRAG** across the two chunking configurations (Configuration A: ~300 tokens vs Configuration B: ~600 tokens).

All analysis is derived directly from actual retrieval logs, cosine similarity scores, and ground-truth comparisons recorded in `question_results.csv` and `recall_comparison.json`.

---

## Failure Case 1: Semantic Duplication & Multi-Chapter Competition (Q11)

### 1. Original Question
> *"What is Flood Plain Zoning and what is its objective in non-structural flood regulation?"*

### 2. Expected Relevant Ground Truth
- **Expected Document:** `NDMA_Floods_Guidelines.pdf`
- **Expected Page:** Page 41 (Executive Summary / Regulation & Enforcement Section 6.1)
- **Ground Truth Passage:**
  > *"Flood Plain Zoning aims at demarcating zones in the flood plains along the rivers based on flood frequencies and regulating development activities within these zones to minimize loss of life and property."*
- **Expected Chunk ID:** `NDMA_Floods_Guidelines_cfg300_c0060` (Config 300) / `NDMA_Floods_Guidelines_cfg600_c0030` (Config 600)

### 3. Actual Top-5 Retrieved Passages (Configuration A: 300 Tokens)

| Rank | Similarity Score | Document | Page | Chunk ID | Content Preview |
|:---:|:---:|---|:---:|---|---|
| **1** | **0.7493** | `NDMA_Floods_Guidelines.pdf` | **Page 78** | `NDMA_Floods_Guidelines_cfg300_c0136` | *"zoning, therefore, aims at determining the locations and the extent of areas for developmental activities in such a fashion that the damage is reduced to a minimum..."* |
| **2** | **0.7415** | `NDMA_Floods_Guidelines.pdf` | **Page 110** | `NDMA_Floods_Guidelines_cfg300_c0204` | *"and disincentives with respect to the central assistance to encourage the states for enactment and enforcement of flood plain zoning regulations. (para 6.2)..."* |
| **3** | **0.7298** | `NDMA_Cyclones_Guidelines.pdf` | **Page 121** | `NDMA_Cyclones_Guidelines_cfg300_c0233` | *"impressing upon the states for necessary follow up action to implement the flood plain zoning approach. even though, a model draft bill was circulated..."* |
| **4** | **0.7235** | `NDMA_Floods_Guidelines.pdf` | **Page 148** | `NDMA_Floods_Guidelines_cfg300_c0268` | *"Model Bill for Flood Plain Zoning: may, by notification in the official gazette, appoint: provided that different dates may be appointed for different provisions..."* |
| **5** | **0.7039** | `NDMA_Floods_Guidelines.pdf` | **Page 79** | `NDMA_Floods_Guidelines_cfg300_c0138` | *"be prohibited in this zone. in area liable to flooding in a 25-year frequency flood, residential buildings could be permitted with certain stipulation..."* |

*Note: The expected passage on Page 41 received a similarity score of 0.6841 and was ranked at position #7.*

### 4. Why Retrieval Failed
The dense vector search did **not** retrieve the expected summary page (Page 41) in the top 5 because of **cross-chapter semantic competition**:
1. The concept *"Flood Plain Zoning"* appears in four separate places in the official guidelines:
   - Page 41: Executive Summary / Policy overview.
   - Page 78: Full dedicated Chapter 6 on Flood Plain Zoning definitions.
   - Page 110: Implementation incentives and enforcement recommendations.
   - Page 148: Annexure containing the legislative Model Draft Bill.
2. The embedding model (`all-MiniLM-L6-v2`) found that the operational passage on **Page 78** had a higher semantic density for *"objective and extent of developmental activities"* (cosine similarity **0.7493**) than the brief introductory mention on Page 41 (**0.6841**).
3. Under a strict page-level ground truth rubric expecting Page 41, this counts as a miss, even though the retrieved passages on Page 78 and Page 110 contain valid and highly relevant answers to the query.

### 5. Architectural Root Cause
- **Semantic Multi-Occurrence / Granularity Mismatch**: The query semantics were better aligned with the deeper substantive body text than the executive summary list.
- **Bi-Encoder Compression**: The bi-encoder compresses entire chunk paragraphs into a single 384-dimensional vector, losing distinctions between "definitional overview" vs "operational regulation details."

### 6. Proposed Technical Improvements
1. **Reciprocal Rank Fusion (RRF) with BM25**: BM25 keyword matching directly targets exact section header matches (`"6.1 Flood Plain Zoning"`), boosting the primary definition chunk.
2. **Cross-Encoder Re-Ranking**: Running a secondary cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`) on top-20 retrieved candidates allows full cross-attention between every token in the query and candidate passages, correctly promoting the foundational definition.
3. **Hierarchical Document Chunking**: Parent-document retrieval or section-header prepending (e.g., prepending `"[Chapter 6: Regulation and Enforcement > 6.1 Flood Plain Zoning]"` to the chunk text) provides explicit structural grounding.

---

## Failure Case 2: Semantic Dilution in Large Chunks (Q05)

### 1. Original Question
> *"What is Rapid Visual Screening (RVS) and what is its role in seismic assessment?"*

### 2. Expected Relevant Ground Truth
- **Expected Document:** `NDMA_Earthquakes_Guidelines.pdf`
- **Expected Page:** Page 21 (Glossary of Technical Terms)
- **Ground Truth Passage:**
  > *"Rapid Visual Screening is a procedure developed for the qualitative assessment of seismic vulnerability of buildings based on visual inspection to identify potential structural deficiencies."*
- **Expected Chunk ID:** `NDMA_Earthquakes_Guidelines_cfg300_c0012` (Config 300) / `NDMA_Earthquakes_Guidelines_cfg600_c0007` (Config 600)

### 3. Actual Top-5 Retrieved Passages (Configuration B: 600 Tokens)

| Rank | Similarity Score | Document | Page | Chunk ID | Content Preview |
|:---:|:---:|---|:---:|---|---|
| **1** | **0.4089** | `NDMA_Earthquakes_Guidelines.pdf` | **Page 51** | `NDMA_Earthquakes_Guidelines_cfg600_c0036` | *"of seismic risk and vulnerability and on structural and non-structural risk reduction measures. the erec (imd) and other knowledge institutions such as the iits..."* |
| **2** | **0.4039** | `NDMA_Cyclones_Guidelines.pdf` | **Page 161** | `NDMA_Cyclones_Guidelines_cfg600_c0160` | *"high resolution (at least 0.5 m interval) coastal land dem mosaics will be developed for micro-scale delineation of cyclone risk, hazard and vulnerability..."* |
| **3** | **0.3930** | `NDMA_Earthquakes_Guidelines.pdf` | **Page 33** | `NDMA_Earthquakes_Guidelines_cfg600_c0017` | *"seismic retrofitting, developing manuals and training professionals. the imd has installed instruments to record strong motion data in some earthquakeprone regions..."* |
| **4** | **0.3711** | `NDMA_Earthquakes_Guidelines.pdf` | **Page 41** | `NDMA_Earthquakes_Guidelines_cfg600_c0025` | *"drawing up the priority list, a cluster approach will be followed for selecting various types of critical lifeline structures and various categories of building types..."* |
| **5** | **0.3542** | `NDMA_Earthquakes_Guidelines.pdf` | **Page 43** | `NDMA_Earthquakes_Guidelines_cfg600_c0027` | *"to develop a consensus on the methodology that should be undertaken for rvs of buildings in india as a part of seismic vulnerability assessment..."* |

*Note: In Configuration A (300 tokens), Q05 was a **SUCCESS** (Rank 2, Score: 0.5635), but in Configuration B (600 tokens), Q05 **FAILED** (Scores dropped to < 0.41, missing top-5).*

### 4. Why Retrieval Failed
1. In **Configuration B (600 tokens)**, each chunk encompasses approximately 600 tokens. The specific definition of *"Rapid Visual Screening"* is concise (~45 words).
2. When creating 600-token chunks, this short definition was concatenated with five other disparate glossary terms on Page 21: *Resilience*, *Risk*, *Risk Assessment*, *Risk Management*, and *Vulnerability*.
3. Because the bi-encoder pools the embeddings of all 600 tokens into one average vector, the specific semantic signal of *"Rapid Visual Screening"* was **diluted** by the surrounding terms.
4. Consequently, the similarity score dropped from **0.5635** (in 300-token chunks) down to **0.3340** (outside the top 5).
5. Generic passages mentioning *"vulnerability assessment"* on Pages 51, 161, and 33 scored higher (~0.40) because they repeatedly mentioned broad risk reduction terms.

### 5. Architectural Root Cause
- **Semantic Dilution & Information Density Loss**: Large chunk sizes (600 tokens) smooth out localized, high-information terminology definitions into a generic topical vector.
- **Model Context Window Truncation**: MiniLM has a native sequence limit of 256 tokens; when encoding 600-token chunks, tokens beyond position 256 are truncated, completely cutting off text appearing in the second half of large chunks!

### 6. Proposed Technical Improvements
1. **BM25 Exact Keyword Weighting**: BM25 computes Inverse Document Frequency (IDF). Since *"RVS"* and *"Rapid Visual Screening"* are rare acronyms across the corpus, BM25 assigns them a massive score boost, instantly ranking the glossary entry #1 regardless of chunk size.
2. **Adaptive / Sentence-Window Chunking**: Rather than fixed 600-token spans, chunk at semantic section boundaries or preserve sentence-level anchors linked to expanded parent context windows.
3. **Hybrid Dense + Sparse Retrieval (Dense + BM25)**: As demonstrated in our hybrid retrieval experiment, combining dense semantic search with sparse keyword search completely resolves acronym and definition dilution.
