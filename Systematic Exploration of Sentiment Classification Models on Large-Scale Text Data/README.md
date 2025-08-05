# A Systematic Exploration of Sentiment Classification Models on Large-Scale Text Data

This project documents a comprehensive investigation into the best methods for sentiment analysis on a massive dataset of 3.6 million Amazon reviews. Rather than settling for a single baseline, I conducted a wide-ranging benchmark, systematically comparing multiple feature engineering techniques and a suite of machine learning models—from classical statistics to deep learning—to identify the most robust and accurate solution.

---

### **1. The Core Problem: Feature Engineering**

The first major challenge was determining the best way to convert unstructured text into meaningful numerical features for the models. I hypothesized that the choice of feature set would be critical, so I tested three distinct approaches:

*   **TF-IDF:** A classical weighting scheme.
*   **Pointwise Mutual Information (PMI):** An information-theoretic approach.
*   **Chi-Square (χ²) Statistical Selection:** A statistical test to find features with the strongest association to the sentiment label.

**Finding:** The **Chi-Square method consistently proved to be the most effective feature selection technique**, particularly for the linear models, leading to the highest overall performance.

---

### **2. The Model Showdown: A Comparative Benchmark**

I then trained and evaluated a diverse set of models on the Chi-Square selected features to understand the performance trade-offs between different algorithmic approaches.

**Key Results Summary:**
The following table summarizes the validation accuracy for the primary models tested.

| Model | Feature Method | Validation Accuracy |
| :--- | :--- | :--- |
| **🏆 Logistic Regression (Tuned, C=12)** | **Chi-Square** | **90.7%** |
| Logistic Regression (Default) | Chi-Square | 90.0% |
| SVM (Linear Kernel) | Chi-Square | 89.0% |
| Logistic Regression | TF-IDF | 89.0% |
| Multinomial Naive Bayes | Chi-Square | 87.4% |
| XGBoost | Chi-Square | 87.0% |
| Random Forest | Chi-Square | 83.5% |

**Finding:** A well-tuned, classic **Logistic Regression model was the surprising champion**. This was a key insight: for this type of high-dimensional, sparse text data, a robust linear model can outperform more complex tree-based ensembles like Random Forest and XGBoost.

---

### **3. The Deep Learning Experiment: A Dive into LSTMs**

To explore the state-of-the-art, I implemented an LSTM network, a type of recurrent neural network designed for sequence data.

*   **Initial Success:** On a similar, pre-cleaned Amazon dataset, the LSTM achieved an impressive **94% validation accuracy**, demonstrating its immense potential.
*   **The Overfitting Challenge:** However, when applied to the original, noisier competition dataset, the model's performance plummeted to ~50% on the test set.

**Finding:** This was a powerful, practical lesson in overfitting. The LSTM, with its millions of parameters, had likely memorized dataset-specific quirks (like deliberate spelling mistakes) rather than learning the generalizable features of language. This highlighted that a more complex model is not always a better one, and its successful application requires careful regularization and tuning, which I would explore in future iterations.

---

### **Conclusion**

This project was a deep dive into the practical realities of applied machine learning. The key takeaway was that a systematic process of experimentation is crucial. The winning combination was not the most complex model, but the one best suited to the data's structure: a **Logistic Regression classifier, carefully tuned, using a feature set selected with the powerful Chi-Square statistical test**, achieving a final validation accuracy of **90.7%**.
