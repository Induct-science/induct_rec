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
