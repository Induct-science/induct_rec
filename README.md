# induct_rec

This repository contains the standalone, decoupled recommendation algorithm for the Induct platform. It is designed to be completely independent from the core database and web server to allow open-source transparency into how paper recommendations and credibility scores are algorithmically computed.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/your-username/induct_rec.git@main
```

## Usage

The algorithm is purely mathematical and operates on standard Python lists and Numpy arrays.

```python
import numpy as np
from induct_rec import build_user_profile_vec, recommend_topk

# 1. Build User Profile Vector
user_papers_data = [
    ("Paper Title 1", "Abstract 1"), 
    ("Paper Title 2", "Abstract 2")
]
keyword_weights = {"machine learning": 3.0, "oceanography": 5.0}

user_vec = build_user_profile_vec(user_papers_data, keyword_weights, alpha=0.7)

# 2. Score Candidates
candidate_vecs = np.random.rand(100, 384) # Example candidates
candidate_ids = list(range(100))

recommendations = recommend_topk(user_vec, candidate_vecs, candidate_ids, k=5)
print(recommendations)
```

## The staged pipeline

`rank_candidates` is what Induct actually runs; `recommend_topk` above is the
older single-pass function, kept as a fallback. The stages, in order:

1. **Similarity** — cosine between the reader's profile vector and each paper.
2. **Gate** — drop everything below `min_similarity`.
3. **Rerank** (optional) — a seam for a cross-encoder; not in use.
4. **Score** — similarity × recency × credibility × the reader's cited-author
   factor.
5. **Select** — reserved slots, with MMR diversification inside them.

### People the reader has cited

The one term that is about the reader rather than the paper. The host passes
`candidate_cited_author_strength` — for each candidate, a number in [0, 1]
saying how strongly this reader has drawn on the people who wrote it — and
the score is multiplied by `1 + cited_author_weight * strength`.

```python
from induct_rec import cited_author_strength, cited_author_factor

cited_author_strength(0)    # 0.0   never cited
cited_author_strength(1)    # 0.29  one of their works
cited_author_strength(10)   # 1.0   ten or more: as much as it ever counts

cited_author_factor(1.0, 0.5)   # 1.5
```

It is applied after the gate, so it reorders papers that already qualified on
relevance and cannot bring an off-topic one in; and it is bounded, so a
reader's closest collaborator is worth at most `1 + weight`. With no
strengths passed, or a weight of 0, the pipeline is exactly what it was.

`cited_author_factor` is exported so a host that stores a ranked feed can
apply the same factor when it serves it, and divide it back out of what it
stores. Who a reader has cited is the host's data; this package never sees a
name.
