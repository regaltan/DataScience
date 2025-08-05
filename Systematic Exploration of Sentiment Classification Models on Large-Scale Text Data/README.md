# A Systematic Benchmark of Sentiment Classification Models for Large-Scale Text Data

This project documents a comprehensive, multi-stage experiment to identify the optimal feature engineering and modeling pipeline for sentiment analysis on a 3.6 million review dataset. The goal was to systematically test a wide variety of techniques to find the highest-performing combination, mirroring a real-world Kaggle competition environment.

---

### **Phase 1: Feature Engineering Evaluation**

The first major challenge was determining the best way to convert unstructured text into meaningful numerical features. I hypothesized that the choice of feature set would be critical, so I tested three distinct approaches:

1.  **TF-IDF:** A classical weighting scheme.
2.  **Pointwise Mutual Information (PMI):** An information-theoretic selection method.
3.  **Chi-Square (χ²) Test:** A statistical test to select features with the strongest association to the sentiment label.

**Key Finding:** Across multiple models, **Chi-Square feature selection consistently proved to be the most effective feature set**, leading to the highest overall accuracy scores as shown in the results below.

---

### **Phase 2: Comprehensive Model Benchmark Results**

A wide suite of models was trained and evaluated on the hold-out validation set. The following table details the full results of this experimentation, ranked by validation accuracy.

| Model | Feature Method | Validation Accuracy |
| :--- | :--- | :--- |
| **🏆 Logistic Regression (Tuned, C=12)** | **Chi-Square** | **90.7%** |
| Logistic Regression (Default) | Chi-Square | 90.0% |
| Logistic Regression | TF-IDF | 89.0% |
| SVM (Linear Kernel) | Chi-Square | 89.0% |
| SVM (Linear Kernel) | TF-IDF | 87.0% |
| Multinomial Naive Bayes | Chi-Square | 87.4% |
| XGBoost | Chi-Square | 87.0% |
| SVM (RBF Kernel) | Chi-Square | 87.0% |
| Logistic Regression | PMI | 87.0% |
| XGBoost | TF-IDF | 85.0% |
| SVM (Linear Kernel) | PMI | 84.0% |
| XGBoost | PMI | 83.0% |
| Decision Tree (Depth=28) | Chi-Square | 75.0% |

**Key Finding:** A well-tuned, classic **Logistic Regression model was the surprising champion**. This was a key insight: for this type of high-dimensional, sparse text data, a robust linear model can outperform more complex tree-based ensembles like Random Forest and XGBoost.

---

### **Phase 3: Deep Learning Exploration (LSTM) & The Engineering Trade-off**

To explore the state-of-the-art, I implemented an LSTM network, a type of recurrent neural network designed for sequence data.

*   **Initial Success on Separate Data:** On a similar, pre-cleaned Amazon dataset, the LSTM achieved an impressive **94% validation accuracy,** demonstrating its immense potential.
*   **The Overfitting Challenge:** However, when applied to this project's dataset, its performance was poor, failing to generalize and effectively collapsing to ~50% test accuracy.

**Analysis of the Result:**
This result provided a powerful, practical lesson in the limitations of complex models. The failure was not inherent to the LSTM architecture itself, but to the training strategy. The model was information-starved, attempting to learn the nuances of English from scratch using a limited vocabulary (`num_words=3500`). This caused it to memorize noise in the training data rather than learning generalizable language patterns.

**The Crucial Insight - Beyond Accuracy:**
Theoretically, a properly configured LSTM with pre-trained embeddings (like GloVe or BERT) could eventually surpass the Logistic Regression model. However, this experiment highlighted a critical engineering principle: **the return on investment.** The Logistic Regression model achieved a >90% accuracy in *minutes* of training time. The LSTM required *hours* and significant computational resources just to produce a poor result.

Achieving a potential 1-2% accuracy gain with the LSTM would require a massive increase in complexity, training time, and cost. For a real-world business application, this is often an unacceptable trade-off.

---

### **Conclusion**

The winning pipeline was a **Tuned Logistic Regression (C=12) using a 5000-feature set selected via the Chi-Square test, achieving 90.7% validation accuracy.** This systematic, multi-stage process of experimentation was crucial in identifying the most effective combination of feature engineering and modeling for this specific, large-scale task.
