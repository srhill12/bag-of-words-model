
# Bag of Words (BoW) Implementation with NLTK and Scikit-learn

This project demonstrates the creation of a Bag of Words (BoW) model using Python's Natural Language Toolkit (NLTK) and Scikit-learn's `CountVectorizer`. The process involves tokenizing sentences, removing punctuation and stopwords, and finally building the BoW model using both custom logic and the `CountVectorizer` from Scikit-learn.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Explanation](#explanation)
- [Output](#output)

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

To run this project, you need to install the required libraries. You can do this by running:

```bash
pip install nltk scikit-learn pandas
```

## Usage

1. **Import the necessary libraries:**

   ```python
   import nltk
   from nltk.corpus import stopwords
   import re
   from sklearn.feature_extraction.text import CountVectorizer
   import pandas as pd
   ```

2. **Download NLTK's 'punkt' package:**

   ```python
   nltk.download('punkt')
   ```

3. **Initialize the stopwords:**

   ```python
   stop_words = set(stopwords.words('english'))
   ```

4. **Define your sentences:**

   ```python
   sentence_1 = "I want to invest for retirement."
   sentence_2 = "Should I invest in mutual funds, or should I invest in stocks?"
   sentence_3 = "I should schedule an appointment with a financial planner."
   ```

5. **Preprocess the sentences by removing punctuation and tokenizing:**

   ```python
   pattern = r'[^a-zA-Z\s]'
   tokens = []

   # Process each sentence
   sentence_1_cleaned = re.sub(pattern, '', sentence_1)
   sentence_1_tokens = nltk.word_tokenize(sentence_1_cleaned.lower())
   tokens.append(sentence_1_tokens)

   sentence_2_cleaned = re.sub(pattern, '', sentence_2)
   sentence_2_tokens = nltk.word_tokenize(sentence_2_cleaned.lower())
   tokens.append(sentence_2_tokens)

   sentence_3_cleaned = re.sub(pattern, '', sentence_3)
   sentence_3_tokens = nltk.word_tokenize(sentence_3_cleaned.lower())
   tokens.append(sentence_3_tokens)
   ```

6. **Remove stopwords:**

   ```python
   filtered_tokens = []
   for token in tokens:
       filtered_token = [word for word in token if word not in stop_words]
       filtered_tokens.append(filtered_token)
   ```

7. **Create a custom Bag of Words:**

   ```python
   bag_of_words = {}
   for i in range(len(filtered_tokens)):
       for word in filtered_tokens[i]:
           if word not in bag_of_words:
               bag_of_words[word] = 0
           bag_of_words[word] += 1
   print(bag_of_words)
   ```

8. **Use `CountVectorizer` to create a BoW model:**

   ```python
   vectorizer = CountVectorizer(stop_words='english')
   bag_of_words = vectorizer.fit_transform([sentence_1, sentence_2, sentence_3])
   bow_df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())
   print(bow_df)
   ```

9. **Output the vocabulary and word occurrences:**

   ```python
   print(bow_df.columns.to_list())
   occurrence = bow_df.sum(axis=0)
   print(occurrence)
   ```

## Explanation

This code takes three sentences related to financial planning and demonstrates the process of creating a Bag of Words model:

1. **Text Preprocessing:** 
   - Sentences are cleaned by removing punctuation using regular expressions.
   - Sentences are then tokenized into words.
   - Stopwords (common words that provide little informational value) are removed.

2. **Custom BoW Creation:** 
   - A dictionary is used to count the occurrences of each word after preprocessing.

3. **Scikit-learn's CountVectorizer:** 
   - `CountVectorizer` is used to automatically create a Bag of Words matrix, where each row represents a sentence and each column represents a word in the vocabulary.

## Output

### Custom Bag of Words Output:

```python
{'want': 1, 'invest': 3, 'retirement': 1, 'mutual': 1, 'funds': 1, 'stocks': 1, 'schedule': 1, 'appointment': 1, 'financial': 1, 'planner': 1}
```

### Scikit-learn CountVectorizer Output:

```python
   appointment  financial  funds  invest  mutual  planner  retirement  schedule  stocks  want
0            0          0      0       1       0        0           1         0       0     1
1            0          0      1       2       1        0           0         0       1     0
2            1          1      0       0       0        1           0         1       0     0
```

### Word Occurrences in the Vocabulary:

```python
appointment    1
financial      1
funds          1
invest         3
mutual         1
planner        1
retirement     1
schedule       1
stocks         1
want           1
dtype: int64
```

This README provides a concise explanation and usage guide for building a Bag of Words model using NLTK and Scikit-learn.
```
