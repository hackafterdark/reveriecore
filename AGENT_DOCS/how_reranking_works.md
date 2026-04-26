# Cross-Encoder Reranking (Stage D)

ReverieCore implements a high-precision reranking stage using [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank). This stage occurs after initial retrieval candidates have been gathered from the vector and graph search stages.

## What is a Reranker?

Initial retrieval (Stage A/B) usually relies on **Bi-Encoders** (Vector search). Bi-Encoders are extremely fast because they represent queries and documents as single vectors and use cosine similarity to find matches. However, they lack "fine-grained" semantic understanding because the query and document vectors are calculated independently.

A **Reranker** (usually a **Cross-Encoder**) takes the query and a candidate document and processes them *together*. This allows the model to attend to the specific relationship between every word in the query and every word in the document.

While much slower than vector search, Cross-Encoders are significantly more accurate. By applying them only to the top candidates (e.g., top 15-20), we get the precision of a Cross-Encoder with the speed of a Bi-Encoder.

## Pipeline Integration

In ReverieCore, the reranker is implemented as a `RetrievalHandler` and fits into the `Retriever` pipeline as **Stage C**:

1.  **Stage A: Vector Discovery**: Quick approximate nearest neighbor search.
2.  **Stage B: Graph Expansion**: Signal-aware 1-hop traversal to find related context.
3.  **Stage C: Reranking (This Stage)**: The `RerankerHandler` takes the top candidates and calculates a high-precision relevance score.
4.  **Stage D: Pruning**: Filtering based on token budget and simple relevance.

Note: Stage B also "prunes" but not for token budget, but for relevance. The reason is because it isn't worth the reranker's compute time.
This similarity threshold can be configured in the settings, default is 0.45.

### Workflow
- Candidates are formatted into a list of passages: `{"id": cid, "text": content_full}`.
- FlashRank executes the `ms-marco-MiniLM-L-12-v2` model (approx. 130MB).
- The original retrieval scores are **overridden** by the reranker's scores.
- The `source` attribute for these candidates is updated to `"reranked"`.

## Model Details

- **Model**: `ms-marco-MiniLM-L-12-v2`
- **Footprint**: ~130MB on disk.
- **Runtime**: ONNX-accelerated (CPU).
- **Latency**: Typically 50ms - 200ms depending on the number of candidates.

## Telemetry & Observability

The Reranker is fully instrumented with OpenTelemetry. You can find the following attributes in Jaeger:

- `retrieval.handler`: `RerankerHandler`
- `rag.retrieval.rerank_candidate_count`: Number of candidates sent to the reranker.
- `rag.retrieval.rerank_latency`: Time taken (in ms) to execute the model.

## Troubleshooting

### Missing Module
If `flashrank` is not installed, the system will log a `DEBUG` message:
`FlashRank not found. Skipping rerank.`
The retrieval will continue using only Stage A/B scores.

### First-Run Delay
The first time the reranker is triggered, it will download the model files to its local cache. You will see an `INFO` log:
`ReverieCore: Initializing Reranker model (ms-marco-MiniLM-L-12-v2)...`
