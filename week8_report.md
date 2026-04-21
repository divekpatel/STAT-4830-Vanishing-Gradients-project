# Week 8 Report

---

## 1. Problem Statement

### **What are you optimizing?**
We continue to optimize the **sequential permutation** ($\pi$) of candidate tracks within a listening session to minimize the cumulative probability of user skips. Given a fixed pool of candidate tracks and the session context observed so far, we seek an ordering that minimizes a numerical cost function derived from skip-prediction probabilities. In Week 8, the underlying predictor was upgraded from Logistic Regression to a **Transformer classifier** trained on compound tokens of behavioral, temporal, and transition features.

### **Why does this problem matter?**
Skip behavior remains the clearest engagement signal for music streaming platforms: higher skip rates correlate with shorter sessions, reduced satisfaction, and lower retention. Ordering is a real product lever (Daily Mix, Radio, Discover Weekly), and improving ordering without changing the underlying catalog directly affects listening flow at scale.

### **Success Metrics & Constraints**
* **Primary Metrics:** Skip precision and recall on held-out sessions, plus Total Predicted Skip Probability (TPSP) for optimized vs. original sequences.
* **Operating Point:** Optimized skip-decision threshold of **0.6879**, selected on validation.
* **Constraints:** The candidate track pool is fixed per session; we optimize ordering only. Validation remains offline (session-based split); no causal A/B evidence.

### **Data & Risks**
We continue to work with the **Spotify Sequential Skip Prediction Challenge (WSDM Cup 2018)** data. This week we merged session logs with full track metadata and audio features, producing **167,880 merged rows** split into **8,000 train / 2,000 test sessions**.
* **Risk:** Class imbalance is most damaging on rare items — the bulk of remaining error concentrates there.
* **Risk:** The model still scores tracks largely independently during the forward pass; transition quality is encoded in input features but not in the output structure.
* **Risk:** Offline evaluation cannot prove causal gains from reordering.

---

## 2. Technical Approach

### **Mathematical Formulation**
We seek a permutation $\pi \in \Pi$ that minimizes:
$$\min_{\pi \in \Pi} \mathcal{J}(\pi) = \sum_{k=11}^{20} \hat{P}(\text{skip}_k \mid \mathbf{x}_{\pi(k)}, \text{Context}_k)$$
where $\hat{P}$ is now produced by a Transformer with a two-head output (SKIP / SONG).

### **Algorithm Choice**
We replaced the Logistic Regression baseline with a **Transformer model operating on compound tokens**. Each token aggregates multiple features (behavioral, temporal, and transition-aware), and the model outputs two heads:
* **SKIP head** — probability the current track is skipped.
* **SONG head** — probability the track is played through (no skip).

Sequencing at inference remains **greedy** for now: at each slot, pick the remaining candidate with the lowest SKIP probability.

### **Feature Engineering**
The Week 8 feature space moves from mostly heuristic signals to a **context-aware** representation:
* **Session position / time context:** `session_progress`, day-of-week, time-of-day, weekend indicator.
* **Transition smoothness:** `delta_*` features (e.g., $Energy_t - Energy_{t-1}$), acoustic similarity between adjacent tracks.
* **Cumulative user behavior:** `cum_skips`, `skip_rate_so_far`, pause counts, tempo deviation from session mean.

### **PyTorch Implementation Strategy**
The Transformer is implemented as an `nn.Module` with learned token embeddings for the compound features, standard encoder blocks, and two linear output heads trained jointly with BCE-style losses. Training uses the Adam optimizer with a held-out validation split to pick the decision threshold.

### **Validation Methods**
We use an 80/20 session-level split (8k train / 2k test). Evaluation reports precision and recall on the held-out test sessions at the optimized threshold, and compares TPSP of the Transformer-reordered sequence against the original Spotify order.

---

## 3. Initial Results

### **Implementation Evidence**
The Transformer training loop converges on the merged 167,880-row dataset. The compound-token pipeline successfully fuses session logs with track metadata and behavioral running statistics.

### **Basic Performance Metrics**
At the optimized threshold of **0.6879**:

| Model | Skip Precision | Skip Recall |
|---|---|---|
| Logistic Regression (Week 6 baseline) | 73% | 63% |
| **Transformer (Week 8)** | **74%** | **74%** |

The precision gain is marginal (+1pp), but the **recall improvement from 63% to 74%** is substantial: the Transformer catches many skip events the baseline missed, which is precisely the class that matters most for ordering.

### **Optimization Gains**
Greedy reordering under the Transformer's skip probabilities continues to produce lower TPSP than the original session order on held-out sessions. The recall gain is the main driver of improved ordering quality.

### **Current Limitations**
* **Acoustic features not fully exploited.** Features like `danceability`, `energy`, `flatness`, and `tempo` exist in the merged data but are not yet included in the token embeddings.
* **Independence in scoring.** Tracks are largely scored independently in the forward pass; transition information lives in input deltas rather than in a sequence decoder.
* **Greedy inference.** No beam search or autoregressive decoder yet, so globally optimal orderings are not guaranteed.
* **Rare-item tail.** Most remaining performance loss concentrates on infrequent tracks/artists.
* **Long-song handling.** Long tracks are treated like short ones despite very different skip dynamics.

---

## 4. Next Steps

### **Immediate Improvements**
* **Expand compound tokens** to include acoustic features (`danceability`, `energy`, `flatness`, `tempo`) in the embedding.
* **Develop a proper sequence algorithm** for playlist generation — move beyond greedy to beam search or an autoregressive decoder using the Transformer itself.
* **Explicit long-song handling**, e.g., duration-normalized position features or a dedicated gating feature.

### **Technical Challenges**
* **Frequency-aware training / class-balanced sampling** to fix the rare-item tail where most error concentrates.
* **Two-tower item embeddings** for better generalization across infrequent tracks.
* **Transition-aware objective** — fold adjacent-track deltas into the loss rather than only into input features.

### **Questions & Learning**
The key lesson this week is that **the upgrade from Logistic Regression to Transformer is mostly a recall story, not a precision story**. This reframes our bottleneck: the model now identifies skip-prone tracks well, and the next gains will come from (a) richer acoustic signal in the tokens, (b) explicit sequence decoding instead of greedy, and (c) handling the rare-item tail where offline metrics still degrade.
