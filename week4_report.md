# Week 4 Report
Project Title: Playlist Ordering Optimization Under Skip Behavior

---

## 1. Problem Statement

### What are you optimizing?
We optimize the ordering of tracks within a playlist to minimize the probability that a user skips songs during a listening session. Given a fixed set of tracks, the objective is to find a permutation that minimizes the expected number of skip events across playlist positions.

### Why does this problem matter?
Skip behavior is a primary engagement signal for music streaming platforms: higher skip rates are strongly associated with shorter sessions, reduced user satisfaction, and lower retention. Playlist ordering is a real product decision (e.g., Daily Mix, Radio, Discover Weekly), and improving ordering without changing the underlying content directly impacts user experience at scale.

### How will you measure success?
Because users cannot be re-exposed to counterfactual playlist orders, success is measured offline using historical listening logs:
- Skip prediction quality (AUC, log loss)
- Ranking-based metrics on reordered playlists (e.g., NDCG, Skip@k)
- Reduction in predicted skip probability under the reordered playlist relative to the original ordering

All results are explicitly framed as offline, non-causal evaluations, consistent with standard recommender system practice.

### What are your constraints?
- No online A/B testing or user retesting
- Historical skip labels were generated under a different ordering
- Playlist ordering is a combinatorial optimization problem
- Limited session context and metadata

### What data do you need?
We require session-level listening data containing:
- Ordered track sequences
- Skip labels or listening duration
- Track metadata (e.g., genre, artist)
- Session boundaries

Our primary dataset is the **Spotify Sequential Skip Prediction (WSDM Cup 2019)** dataset, which includes skip labels for future tracks in a session and supports offline evaluation of reordering policies.

### What could go wrong?
- Skip probability estimates may be poorly calibrated and compound errors during reordering
- Position bias may dominate true preference signals
- Offline metrics may overestimate real-world gains
- Simple models may fail to capture long-range session dependencies

---

## 2. Technical Approach

### Two-Model Pipeline Overview
The project is structured as a two-stage pipeline:
1. **Skip Prediction Model:** Estimate the probability that a track will be skipped given its context.
2. **Playlist Reordering Model:** Use predicted skip probabilities to reorder tracks to minimize expected skips.

This separation mirrors real-world recommender pipelines and simplifies implementation, testing, and validation.

### Mathematical Formulation
Let a playlist consist of tracks $ \{t_1, \dots, t_n\}$. We seek a permutation $ \pi $ that minimizes expected skip events:

$
\min_{\pi} \sum_{k=1}^{n} \mathbb{E}[\text{skip}(t_{\pi(k)} \mid \mathcal{C}_k)]
$

where $ \mathcal{C}_k $ denotes session context, including previous tracks and position.

### Baseline Skip Prediction Model (Week 4 Focus)
We begin with a frequency-based probabilistic baseline, motivated by interpretability and robustness.

**Transition-based baseline:**

$
\\ P(\text{skip} \mid g_i, g_j) = \frac{\text{count}(g_i \text{ skipped after } g_j)}{\text{count}(g_i \text{ follows } g_j)}
$

where:
- $ g_i $ is the genre of the current track
- $ g_j $ is the genre of the immediately preceding track

We also evaluate a relaxed formulation:

$
P(\text{skip} \mid g_i, \exists g_j \text{ earlier in session})
$

These baselines capture empirical transition effects without learning parameters.

### Extended Skip Prediction Model (Planned)
- Logistic regression implemented in PyTorch
- Features include position, track genre, artist overlap, and genre transitions
- Trained using SGD and compared against the empirical baseline

### Playlist Reordering Strategy
For Week 4, we implement a greedy reordering algorithm:
1. Compute predicted skip probabilities for each track
2. Sort tracks in ascending order of predicted skip probability
3. Compare reordered playlists against original orderings

Planned extensions include transition-aware costs and Frank–Wolfe relaxations over permutation matrices.

### PyTorch Implementation Strategy
- Skip predictor implemented as a standalone PyTorch module
- Explicit logging of losses, gradients, and convergence behavior
- CPU-based training with small batches for rapid iteration

### Validation Methods
Due to the absence of counterfactual user feedback:
- Skip prediction is evaluated using AUC and log loss
- Reordering is evaluated using ranking metrics (NDCG, Skip@k)
- Predicted skip reduction is reported alongside observed historical labels
- Limitations are explicitly documented

### Resource Requirements
- Dataset fits in local memory (<10GB)
- CPU probably sufficient, GPU optional
- Libraries: PyTorch, NumPy, Pandas

---

## 3. Initial Results

### Evidence the implementation works
- Session data successfully loaded and parsed
- Genre transition frequencies computed correctly
- Conditional skip probabilities produced without errors
- Reordered playlists generated end-to-end

### Basic performance metrics
- Baseline model shows meaningful variation in skip probability across genre transitions
- Greedy reordering reduces average predicted skip probability on held-out sessions

### Test case results
- Synthetic sessions validate probability computations
- Edge cases (single-track sessions) handled correctly

### Current limitations
- No position bias correction yet
- No personalization across users
- Reordering ignores transition smoothness

### Resource usage
- Runtime per experiment: under 1 minute
- Memory usage: under 2GB

### Unexpected challenges
- Sparse genre transitions for rare genres
- Noisy skip signals for short tracks

---

## 4. Next Steps

### Immediate improvements needed
- Add explicit position bias features
- Implement logistic regression skip predictor in PyTorch
- Evaluate probability calibration

### Technical challenges to address
- Offline validation without causal guarantees
- Transition-aware ordering beyond greedy sorting

### Questions needing help
- Best offline metrics for evaluating playlist ordering
- How to rigorously control for position bias

### Alternative approaches to try
- Pairwise ranking losses
- Session-level survival modeling

### What we have learned so far
- Simple empirical baselines are strong and informative
- Validation design is as important as model complexity
- Separating prediction from optimization clarifies system structure
