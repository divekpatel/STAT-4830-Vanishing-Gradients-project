# Week 6 Report

---

## 1. Problem Statement

### **What are you optimizing?**
We are optimizing the **sequential permutation** ($\pi$) of music tracks within the second half of a listening session to minimize the cumulative probability of user skips. Given a pool of $N$ candidate tracks and a fixed session history, we seek to find an ordering that minimizes a numerical cost function derived from skip-prediction probabilities.

### **Why does this problem matter?**
User retention on streaming platforms is driven by "listening flow". High skip rates indicate a break in this flow and often lead to session abandonment. By shifting from static recommendation (what to play) to sequential optimization (how to order), we can proactively maintain user engagement.

### **Success Metrics & Constraints**
* **Primary Metric:** Reduction in Total Predicted Skip Probability (TPSP) for optimized sequences vs. original sequences.
* **Success Threshold:** Outperforming the "Spotify Original Order" by at least 5% on held-out test sessions.
* **Constraints:** The track pool is fixed for each session; we are only optimizing the **ordering** of available tracks.

### **Data & Risks**
We utilize the **Spotify Sequential Skip Prediction Challenge** mini-subset, consisting of 10,000 sessions with user behavior logs and 18 acoustic features per track.  
* **Risk:** Predicted probabilities may be **miscalibrated**, causing the optimizer to exploit "false" zeros.
* **Risk:** **Position Bias**—the tendency for users to skip more at the end of a session regardless of the song—may mask true musical preferences.

---

## 2. Technical Approach

### **Mathematical Formulation**
We seek a permutation $\pi \in \Pi$ that minimizes the objective:
$$\min_{\pi \in \Pi} \mathcal{J}(\pi) = \sum_{k=11}^{20} \hat{P}(\text{skip}_k \mid \mathbf{x}_{\pi(k)}, \text{Context}_k)$$
Where $\hat{P}$ is the output of a PyTorch Logistic Regression model.

### **Algorithm Choice**
We implemented **Logistic Regression** as our baseline predictor because it is computationally efficient and provides interpretable feature weights. For the sequencing task, we currently employ a **Greedy Search Inference**. At each step, the algorithm selects the track with the lowest predicted skip cost from the remaining candidate pool.

### **PyTorch Implementation Strategy**
The model is built as an `nn.Module` using a `Sigmoid` output layer to convert logits into probabilities. We use **Binary Cross-Entropy (BCE) Loss** and the **Adam optimizer** for training. 

### **Validation Methods**
We split the data 80/20 into training and testing sets. Validation involves taking the context of a held-out session, calculating the TPSP for its original order, and comparing it to the TPSP of our optimized reordering.

---

## 3. Initial Results

### **Implementation Evidence**
Our PyTorch training loop successfully converges on the mini-subset dataset. The model successfully handles the complex merge of sequential logs and static track metadata.

### **Basic Performance Metrics**
* **Binary Skip Accuracy:** ~68%.
* **Strongest Feature Signal:** `hist_user_behavior_reason_start` is the most informative feature; tracks starting via `trackdone` (natural finish) have a skip rate of ~37.6%, while those started via `fwdbtn` (manual skip) jump to ~85.6%.

### **Optimization Gains**
Initial tests show that **Greedy Optimization** reduces the Total Predicted Skip Probability by **~12.4%** compared to a randomized sequence.

### **Current Limitations**
* **Resource Usage:** Memory usage in Google Colab is currently ~2.4GB RAM for processing the mini-subset.
* **Unexpected Challenge:** The model currently treats each skip probability as **independent** of the *transition* from the previous song, ignoring "musical flow" (e.g., abrupt changes in tempo).

---

## 4. Next Steps

### **Immediate Improvements**
* **Position Bias Features:** Add explicit features for the track's position in the session to disentangle fatigue from preference.
* **Calibration Evaluation:** Implement a **Reliability Diagram** to ensure our costs ($P$) reflect real-world frequencies.

### **Technical Challenges**
* **Transition-Aware Ordering:** We need to incorporate "delta" features (e.g., $Energy_{t} - Energy_{t-1}$) to model the cost of abrupt transitions.
* **Pairwise Ranking:** We plan to explore **RankNet** or **LambdaRank** losses to prioritize the *relative order* of songs over absolute skip prediction.

### **Questions & Learning**
We have learned that **Start Reason** is the single most critical variable in determining current user intent. We are now investigating how to move from **Greedy Search** to **Beam Search** to find more global sequence optima.
