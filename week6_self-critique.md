# Self-Critique: Week 6 Baseline & Optimization Loop
---

## **1. OBSERVE**
The Logistic Regression model successfully achieves a baseline accuracy of ~68% on the binary skip prediction task. However, the system currently lacks musical "flow" logic and is highly dependent on behavioral features rather than acoustic transitions.

---

## **2. ORIENT**

### **Strengths**
* **Numerical Foundation:** We successfully transitioned from frequency-based counts to a differentiable PyTorch Logistic Regression model.
* **Feature Engineering:** Correctly identified `hist_user_behavior_reason_start` as the primary signal for current user intent.
* **Working Loop:** The inference engine successfully executes a full reordering of candidate tracks to minimize a defined objective function.

### **Areas for Improvement**
* **Calibration:** The current Sigmoid output provides a "score," but we have not yet verified if it is a well-calibrated probability (e.g., via a Reliability Diagram).
* **Musical Continuity:** The objective function treats tracks as independent events, failing to penalize jarring transitions in energy or tempo.
* **Search Strategy:** Greedy search is prone to local optima; we are missing a more global search mechanism like Beam Search.

### **Critical Risks & Assumptions**
* **Assumption:** We assume that minimizing the sum of predicted skip probabilities is the optimal way to maximize session length. 
* **Risk:** There is a risk of "Algorithm Aversion" if our optimized playlists lack the natural variance users expect, even if they have lower skip costs.

---

## **3. DECIDE: Concrete Next Actions**

* **Action 1 (Model):** Implement **Position Encoding** as an explicit feature to disentangle user fatigue from song-specific preference.
* **Action 2 (Optimization):** Develop a **Transition Cost Matrix** based on acoustic deltas (e.g., changes in BPM or valence) to add as a penalty term to the objective function.
* **Action 3 (Inference):** Implement a **Beam Search** algorithm with a width of $k=5$ to explore more optimal permutations than the current greedy approach.

---

## **4. ACT: Resource Needs**

* **Knowledge Need:** Look into **Platt Scaling** and **Isotonic Regression** for probability calibration in PyTorch.
* **Technical Need:** We will utilize the `torchaudio` or `librosa` documentation to better understand how to weight acoustic transitions.
