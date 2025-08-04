# NLP and Statistical Analysis of Amazon Reviews

This project is an end-to-end analysis of a corpus of 30,000 Amazon reviews. The goal is to move beyond simple keyword counting and apply foundational data mining and NLP techniques to extract deeper, statistically significant insights from unstructured text data.

The analysis is divided into three core sections:
1.  **Word Association Mining:** Using Pointwise Mutual Information (PMI) to discover meaningful word pairings that co-occur more often than by random chance.
2.  **Feature Selection for Sentiment:** Applying the Chi-Square (χ²) statistical test to identify the words most predictive of a positive or negative review.
3.  **Algorithmic Spell Correction:** Implementing and comparing string similarity algorithms to build a functional spell corrector.

---

### Dataset

The analysis was performed on the `amazon_reviews.txt` corpus, a dataset containing 30,000 user reviews (15,091 positive, 14,909 negative), each labeled with a binary sentiment. This dataset was provided as part of the Data Mining course at UNC-Chapel Hill.

*(Note: The raw dataset is not included in this repository to respect its original distribution rights.)*

---

### Section 1: Word Association Mining with Pointwise Mutual Information (PMI)

**The Goal:**
To identify which pairs of words (e.g., "stainless", "steel") appear together in close proximity more frequently than would be expected by chance. This helps uncover semantic relationships and common phrases within the review text.

**The Method:**
Pointwise Mutual Information (PMI) is a measure from information theory used to find associations. A high PMI score indicates that two words are strongly associated. The process involved cleaning and tokenizing the text, calculating individual and pair frequencies within a 5-word window, and applying the PMI formula.

**Key Results:**
After filtering for pairs appearing at least 50 times, the analysis revealed the top word associations by PMI. Here are the top 10:

```
Word-Pair: ('blah', 'blah'), PMI: 9.949
Word-Pair: ('sci', 'fi'), PMI: 9.696
Word-Pair: ('hip', 'hop'), PMI: 9.670
Word-Pair: ('harry', 'potter'), PMI: 9.626
Word-Pair: ('stainless', 'steel'), PMI: 9.428
Word-Pair: ('blu', 'ray'), PMI: 8.925
Word-Pair: ('buyer', 'beware'), PMI: 8.688
Word-Pair: ('windows', 'xp'), PMI: 8.453
Word-Pair: ('tech', 'support'), PMI: 7.970
Word-Pair: ('web', 'site'), PMI: 7.969
```
*The results successfully identified logically-linked word pairs, demonstrating the effectiveness of PMI in capturing strong semantic connections beyond simple co-occurrence frequency.*

---

### Section 2: Feature Selection using Chi-Square (χ²) for Sentiment Analysis

**The Goal:**
To determine which individual words are the most powerful predictors of review sentiment. This is a critical step in building any sentiment analysis model, as it helps identify the most informative features.

**The Method:**
The Chi-Square (χ²) test of independence was used to find significant associations between word presence and sentiment labels (positive/negative). A high Chi-Square value suggests the word is a strong indicator of sentiment.

**Key Results:**
The analysis produced a ranked list of the most sentiment-bearing words. The following are the top 5 words most predictive of a positive or negative sentiment based on their χ² score:

**Top 5 Most Predictive Positive Words:**
*(Identified by having a high χ² score and higher frequency in positive reviews)*
```
Word: great, Chi-Square: 2259.2
Word: love, Chi-Square: 769.5
Word: best, Chi-Square: 757.5
Word: excellent, Chi-Square: 646.8
Word: wonderful, Chi-Square: 442.9
```

**Top 5 Most Predictive Negative Words:**
*(Identified by having a high χ² score and higher frequency in negative reviews)*
```Word: waste, Chi-Square: 1300.8
Word: money, Chi-Square: 1204.9  (Note: Often follows 'waste of')
Word: poor, Chi-Square: 668.0
Word: worst, Chi-Square: 663.6
Word: disappointed, Chi-Square: 636.8
```
*This statistical feature selection is foundational for building accurate and, importantly, interpretable machine learning models.*
---

### Section 3: Spell Correction using N-gram Jaccard Similarity

**The Goal:**
To build a computationally efficient spell corrector and evaluate its performance against a baseline. Given a potentially misspelled word, the system must return the most likely correct words from a dictionary.

**The Method:**
This section explores the classic trade-off between computational speed and accuracy.
1.  **The "Gold Standard" (Baseline):** Levenshtein Edit Distance is highly accurate for measuring string similarity but is computationally expensive, with a time complexity of O(m*n).
2.  **The "Efficient Approximation":** This implementation uses a faster, set-based approach. Each word is converted into a set of overlapping character n-grams. The **Jaccard Similarity** between the sets is then calculated, which has a much faster time complexity of O(m+n).

The experiment tested n-gram lengths of 2, 3, 4, and 5 to find the optimal balance for this task.

**Key Results:**
The model was tested on a list of challenging, misspelled words. The following shows a curated example for the input "artifitial inteligwnse," comparing the Jaccard method's top suggestions against the Edit Distance baseline.

**Input String:** `artifitial inteligwnse`

| N-gram Size | Top Jaccard Suggestion | Similarity Score |
| :--- | :--- | :--- |
| **n=2** | `artificial intelligence` | 0.5556 |
| **n=3** | `artificial intelligence` | 0.4138 |
| **n=4** | `artificial intelligence` | 0.3000 |
| **n=5** | `artificial intelligence` | 0.2333 |

**Edit Distance Baseline Results (most accurate):**
```
1. artificial intelligence (distance: 4)
2. artificial intelligences (distance: 5)
3. artificial life (distance: 9)
```

**Interpretation:**
The results clearly demonstrate the viability of the n-gram Jaccard approach. Across all tested n-gram sizes, the model successfully identified the correct phrase, `artificial intelligence`, as the top match. The similarity scores decrease as 'n' increases, which is expected as longer n-grams create more specific and thus smaller sets. This experiment shows that for a task like spell correction, an efficient approximation can yield the same top result as the more computationally expensive gold standard, a crucial insight for building scalable NLP systems.
