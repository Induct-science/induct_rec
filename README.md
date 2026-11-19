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

1. **Similarity** — cosine between the reader's profile vector and each paper,
   or the nearest of the reader's interests when that is closer.
2. **Gate** — drop everything below `min_similarity`.
3. **Rerank** (optional) — a seam for a cross-encoder; not in use.
4. **Score** — similarity × recency × credibility × the reader's cited-author
   factor.
5. **Select** — reserved slots, with MMR diversification inside them, and the
   open positions dealt between the reader's interests.

### A reader's interests

One vector — the mean of everything a reader has written — describes somebody
with one field. For somebody with two it is a point between them, and what is
nearest that point resembles both fields a little and neither closely. So a
reader may also have up to three **interests**: groups of their own papers,
each with a centre and a share of their work.

```python
from induct_rec import build_user_interests, rank_candidates

interests = build_user_interests(papers, paper_vecs=stored_vectors)
# [{"vec": ..., "share": 0.67, "papers": [0, 1, 4, ...]},
#  {"vec": ..., "share": 0.33, "papers": [2, 3, ...]}]

if len(interests) > 1:
    ranked = rank_candidates(
        profile_vec, candidate_vecs, candidate_ids,
        interest_vecs=[i["vec"] for i in interests],
        interest_shares=[i["share"] for i in interests],
    )
```

**Grouping** (`cluster_interests`) is average-linkage agglomeration: every
paper starts alone, and the two closest groups are joined until no two are as
close as a mean cosine of 0.30. A group under 15% of the reader's work, or
under three papers, is joined to the one it is nearest; and there are never
more than three. No random start, no paper discarded. On 600 real readers
with six or more papers those settings give 80% one interest, 16% two and 4%
three.

**Relevance** (`reader_relevance`) becomes the cosine to the nearest interest
*or to the profile vector, whichever is larger*. Keeping the profile vector in
the maximum means relevance only ever rises when a reader is given interests:
no paper that passed the gate before fails it now, and a paper that bridges
two of the reader's fields keeps the relevance the mean gave it.

**Selection** deals the open positions between interests by share. Each
position goes to the interest furthest behind its share so far and is filled
by that interest's best remaining paper, so every interest stays within one
paper of its share at every point down the list. Without that, the fields
compete on one score and the one with the better-cited literature takes the
page, whatever the reader's own balance is. A reserved slot is filled as it
always was, and the paper in it counts toward its own interest.

The score formula is untouched. With no interests passed the pipeline is
exactly what it was, and a host should pass none for a reader who has only
one — that reader's one interest *is* their profile vector.

An interest is made of papers only. Blending the reader's keywords in, as
`build_user_profile_vec` does for the single vector, was measured and
dropped: a reader's heaviest keywords are the broad ones, and they pulled
each interest back toward the middle of the discipline.

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
