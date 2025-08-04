# NLP and Statistical Analysis of Amazon Reviews

This project is an end-to-end analysis of a corpus of 10,000 Amazon reviews. The goal is to move beyond simple keyword counting and apply foundational data mining and NLP techniques to extract deeper, statistically significant insights from unstructured text data.

The analysis is divided into three core sections:
1.  **Word Association Mining:** Using Pointwise Mutual Information (PMI) to discover meaningful word pairings that co-occur more often than by random chance.
2.  **Feature Selection for Sentiment:** Applying the Chi-Square (χ²) statistical test to identify the words most predictive of a positive or negative review.
3.  **Algorithmic Spell Correction:** Implementing and comparing string similarity algorithms (N-gram Jaccard Similarity vs. Edit Distance) to build a functional spell corrector.

---

### Section 1: Word Association Mining with Pointwise Mutual Information (PMI)

**The Goal:**
To identify which pairs of words (e.g., "deep", "dish") appear together in close proximity more frequently than would be expected by chance alone. This helps uncover semantic relationships and common phrases within the review text.

**The Method:**
Pointwise Mutual Information (PMI) is a measure from information theory used to find associations. A high PMI score indicates that two words are strongly associated. The process involved:
1.  Cleaning and tokenizing the raw review text.
2.  Calculating the frequency of individual words (unigrams).
3.  Calculating the frequency of ordered word pairs within a sliding text window (size=5).
4.  Applying the PMI formula to quantify the strength of association for each pair.

**Interpretation:**
The results successfully identified logically-linked word pairs. Interestingly, the highest PMI scores didn't always come from the most frequent pairs, highlighting how PMI can uncover strong but less common associations, which simple frequency counts would miss.

---

### Section 2: Feature Selection using Chi-Square (χ²) for Sentiment Analysis

**The Goal:**
To determine which individual words are the most powerful predictors of review sentiment (positive vs. negative). This is a critical step in building any sentiment analysis model, as it helps identify the most informative features and reduce noise.

**The Method:**
The Chi-Square (χ²) test of independence is a statistical tool used to determine if there is a significant association between two categorical variables. Here, the variables are:
1.  The presence or absence of a specific word.
2.  The sentiment label (positive/negative) of the review.

A high Chi-Square value suggests a strong dependence, meaning the presence of the word is a strong indicator of the review's sentiment. I constructed a 2x2 contingency table for each word and calculated its χ² statistic.

**Interpretation:**
The analysis produced a ranked list of the most sentiment-bearing words. While many were predictable ("waste", "disappointed" vs. "great", "excellent"), some were more subtle. This process is foundational for understanding and mitigating bias in text data, a key ethical concern in modern AI systems.

---

### Section 3: Spell Correction using N-gram Jaccard Similarity

**The Goal:**
To build a computationally efficient spell corrector. Given a potentially misspelled word, the system must return the most likely correct words from a dictionary.

**The Method:**
While Edit Distance is highly accurate, it's computationally expensive. This implementation uses a faster, set-based approach:
1.  Each word (both the input and dictionary words) is broken down into a set of overlapping character n-grams (e.g., "hello" with n=3 -> {'hel', 'ell', 'llo'}).
2.  The **Jaccard Similarity** is calculated between the input word's n-gram set and each dictionary word's set.
3.  The dictionary words with the highest Jaccard score are returned as the most likely corrections.

**Interpretation:**
I experimented with different n-gram lengths (2, 3, 4, 5). The results showed that trigrams (n=3) and 4-grams offered a robust balance between accuracy and specificity, providing a strong approximation of the more costly Edit Distance method. This demonstrates a core engineering trade-off between performance and accuracy.
