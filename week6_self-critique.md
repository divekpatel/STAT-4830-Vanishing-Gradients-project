# Self-Critique (OODA Loop)

## OBSERVE
- **Report**: The problem statement is clear, but the mathematical formulation for the optimization part is still theoretical and not yet implemented in code. The focus has been entirely on the prediction baseline.
- **Code**: The heuristic baseline is effective and runs quickly. However, the evaluation metric (Accuracy) might be misleading if classes are imbalanced (though 66/34 is manageable). We haven't yet used the track audio features (`acousticness`, etc.) which was a core goal.
- **Results**: The baseline accuracy is likely high solely due to the `start_reason` feature, masking potential difficulties in predicting skips for "passive" listening sessions.

## ORIENT
**Strengths:**
1.  **Strong Baseline**: We established a rigorous "ground truth" (if user hits `fwdbtn`, they will likely skip again) that any complex model must beat.
2.  **Clear Metric**: Moving from `skip_1`/`skip_2` to a consolidated `skipped` boolean simplifies the optimization target.
3.  **Data Pipeline**: The merge and cleaning process is robust and handles the mini-dataset well.

**Areas for Improvement:**
1.  **No Machine Learning**: The current "model" is just a hard-coded heuristic. It doesn't "learn" weights.
2.  **Ignored Features**: We completely ignored the `tf_mini` (audio features) data in the prediction logic.
3.  **Evaluation Rigor**: We only used Accuracy. We should calculate AUC-ROC and Log Loss to better understand probabilistic confidence.

**Critical Risks/Assumptions:**
- **Assumption**: We assume `start_reason` is always available at inference time. In a real re-ordering scenario, we might not know *how* a track starts until it happens.
- **Risk**: The model might fail to generalize to new users or sessions where behavior is less explicit (e.g., purely radio-mode listening).

## DECIDE
**Concrete Next Actions:**
1.  **Implement Logistic Regression**: Replace the `groupby` heuristic with a `sklearn.linear_model.LogisticRegression` model that uses both behavioral one-hot encoded features AND audio features.
2.  **Feature Engineering**: Normalize/Scale audio features (0-1 range) to make them compatible with linear models.
3.  **Expand Metrics**: Add `roc_auc_score` and `log_loss` to the evaluation pipeline.

## ACT
**Resource Needs:**
- **Tools**: standard `scikit-learn` libraries are sufficient for the next step. PyTorch will be needed later for the custom loss optimization.
- **Blockers**: None currently. The dataset fits in memory.
- **Knowledge**: Need to verify if we should use `StandardScaler` or `MinMaxScaler` for the audio features distributions.
