# Self-Critique: Week 8 Transformer Upgrade
---

## **1. OBSERVE**
The Transformer classifier with compound-token inputs reaches **74% precision and 74% recall** on the binary skip-prediction task, compared to the Week 6 Logistic Regression baseline of 73% / 63%. The jump is concentrated in recall — the model now catches substantially more true skip events — while precision barely moved. The decision threshold settled at **0.6879** on validation. The inference path, however, is still greedy, and scoring still treats candidate tracks largely as independent events despite the richer context-aware feature space.

---

## **2. ORIENT**

### **Strengths**
* **Architecture upgrade landed cleanly:** The Transformer trains to convergence on the 167,880-row merged dataset and produces a usable two-head (SKIP / SONG) output.
* **Meaningful recall gain:** Moving from 63% → 74% recall is the most important number this week — skip events are what drive ordering decisions, and we now surface more of them.
* **Richer feature space:** Compound tokens successfully fuse behavioral history (`cum_skips`, `skip_rate_so_far`), temporal context (`session_progress`, day/time/weekend), and transition deltas into a single representation.
* **Disciplined threshold selection:** We tuned the decision threshold on validation (0.6879) rather than defaulting to 0.5.

### **Areas for Improvement**
* **Precision barely improved** (+1pp). A transformer-sized architectural jump that yields only 1pp of precision suggests either (a) we are near the ceiling of the current feature set, or (b) class imbalance is capping us.
* **Acoustic features are still on the bench.** `danceability`, `energy`, `flatness`, and `tempo` exist in the merged data but are not yet inside the token embeddings — a clear, cheap source of remaining signal.
* **Sequence is still scored piecewise.** Despite using a Transformer, we evaluate each candidate in isolation at inference. The architecture can do more than we are asking of it.
* **Greedy inference persists.** Beam search and autoregressive decoding were called out in Week 6 and are still not implemented.
* **Rare-item tail is unaddressed.** Most of the remaining loss sits on infrequent tracks, and we have not yet introduced class-balanced sampling or frequency-aware weighting.

### **Critical Risks & Assumptions**
* **Assumption:** The 0.6879 threshold selected on validation generalizes to new sessions. We have not yet checked threshold stability across session types (e.g., short vs. long sessions, weekday vs. weekend).
* **Risk:** The recall gain may partly reflect the optimized threshold rather than pure model capability — we should confirm the gain survives at a shared operating point.
* **Risk:** Long songs are treated like short ones; their skip dynamics are different and this is currently a silent failure mode.
* **Risk:** Offline gains continue to lack causal grounding — there is still no A/B evidence that reordered playlists reduce real-world skips.

---

## **3. DECIDE: Concrete Next Actions**

* **Action 1 (Features):** Add `danceability`, `energy`, `flatness`, and `tempo` into the compound token embeddings. This is the lowest-effort, highest-expected-value next step.
* **Action 2 (Imbalance):** Introduce **class-balanced sampling** or a **frequency-aware loss weighting** to pull performance up on rare items, where most of the remaining loss lives.
* **Action 3 (Inference):** Replace greedy ordering with **beam search** (k=5) or an **autoregressive decoder** head on the Transformer, so sequencing uses the architecture we already paid for.
* **Action 4 (Long songs):** Add explicit **duration gating** or a long-song indicator feature, and evaluate skip metrics stratified by track length.
* **Action 5 (Validation):** Compare Transformer vs. Logistic Regression at a **matched threshold** to confirm the recall gain is not just a threshold artifact.

---

## **4. ACT: Resource Needs**

* **Knowledge Need:** Review **two-tower item embedding** literature for handling the rare-item tail, and revisit class-imbalance techniques (focal loss, weighted BCE, balanced sampling).
* **Technical Need:** Extend the feature pipeline to merge the remaining audio features into the compound tokens, and prototype `torch.nn.TransformerDecoder` for sequence generation.
* **Evaluation Need:** Build a **per-threshold comparison plot** (precision/recall vs. threshold) for both models, plus a **stratified metrics table** (by session length, track duration, item frequency) so we can diagnose where gains actually come from.
