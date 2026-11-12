# IMDb Movie Review Sentiment Analysis

## Project Overview

This project uses natural language processing (NLP) and supervised machine learning to classify IMDb movie reviews as **positive** or **negative**. It combines text preprocessing, term frequency-inverse document frequency (TF-IDF) features, and a Multinomial Naive Bayes classifier. It also compares performance using different numbers of labelled training reviews.

The project was developed for the Natural Language Processing module at IU International University of Applied Sciences.

## Dataset and Experimental Design

This project uses the Large Movie Review Dataset (IMDb) introduced by Maas et al. (2011).

The dataset contains 50,000 labelled movie reviews, divided into 25,000 training reviews and 25,000 test reviews.

### Dataset Sources

- **Original Stanford source:** https://ai.stanford.edu/~amaas/data/sentiment/
- **Alternative Zenodo download:** https://zenodo.org/records/7929635

The Zenodo dataset was verified against the original dataset used in this project. All 50,000 review files matched by filename and text content.

### Dataset Setup

1. Download `large_movie_review_dataset.zip` from Zenodo.
2. Extract the ZIP archive.
3. Copy the `train` and `test` folders into `data/aclImdb/`
4. Run the Jupyter notebook according to the project instructions.

**Note:** The dataset is not included in the GitHub repository.
**Methodological limitation:** An earlier exploratory notebook accessed a small sample of the official test partition during development. Although the present notebook separates training, validation and final testing, the test partition was not strictly untouched throughout the entire project history. This should be considered when interpreting the final test result.

## Methodology

### Text Preprocessing

Reviews were lowercased and IMDb HTML line-break tags (`<br />`) were replaced with spaces. NLTK word tokenisation and part-of-speech (POS) tagging were used, followed by stop-word filtering and WordNet lemmatisation. Negation terms including `not`, `no`, `never`, and `n't` were preserved. Tokens without alphanumeric characters were removed, and remaining tokens were joined into cleaned text.

### TF-IDF Feature Extraction

`TfidfVectorizer(max_features=5000)` converted each review into a numerical vector, weighting terms according to their occurrence in the review and their rarity across the training corpus. A **separate vectoriser was fitted only on each training subset**, then applied to the validation reviews. The selected training vectoriser was reused for the final test set, avoiding vocabulary and IDF leakage from validation and test data.

### Classification

A `MultinomialNB` classifier was fitted on each training subset's TF-IDF vectors. Accuracy, macro-averaged precision, recall, and F1-score were calculated on the validation set.

## Recorded Results

| Training reviews | Validation accuracy | Macro precision | Macro recall | Macro F1 |
|---:|---:|---:|---:|---:|
| 1,000 | 79.98% | 80.74% | 79.98% | 79.86% |
| 5,000 | 83.36% | 83.37% | 83.36% | 83.36% |
| 10,000 | 84.30% | 84.30% | 84.30% | 84.30% |

The selected model's **recorded final test accuracy was 83.23%** on 25,000 reviews. The notebook's classification report shows negative-class precision/recall/F1 of 0.82/0.86/0.84 and positive-class precision/recall/F1 of 0.85/0.81/0.83 (rounded to two decimals).

These results are recorded in the notebook and report. The project author has confirmed that a fresh run reproduced the validation and final test accuracy values and regenerated the six expected output files. The cleaned notebook uses sorted dataset filenames; an earlier exploratory notebook may have used a different input ordering, so results depend on the precise data preparation procedure.

## Project Structure


```text
imdb-sentiment-analysis/
├── movie_review_sentiment_analysis.ipynb
├── README.md
├── .gitignore
├── requirements.txt
├── Project_Report.docx
└── data/                            # Local, not uploaded
    ├── aclImdb/                     # Downloaded IMDb dataset
    │   ├── train/{pos,neg}/
    │   └── test/{pos,neg}/
    └── saved/                       # Generated when the notebook runs
        ├── validation_results.csv
        ├── training_size_vs_accuracy.png
        ├── final_confusion_matrix.png
        ├── processed_test_df.pkl
        ├── sentiment_model.pkl
        └── tfidf_vectorizer.pkl
```


The `data/` directory is excluded from Git by `.gitignore`. The notebook creates `data/saved/` automatically and writes the six listed outputs. The notebook **does not currently generate** `experiment_results.csv`.

## Setup

1. Install Python and Jupyter Notebook.
2. Install dependencies using the included `requirements.txt`:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Download the [IMDb Large Movie Review Dataset](https://zenodo.org/records/7929635) and extract the `aclImdb` folder to `data/aclImdb/`, preserving `train/pos`, `train/neg`, `test/pos`, and `test/neg`.
4. From the project root, launch Jupyter (`jupyter notebook`) and open `movie_review_sentiment_analysis.ipynb`.
5. Run the cells **in order from a fresh kernel**. The notebook attempts to download the required NLTK resources (`punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`, `wordnet`). Internet access may be required on the first run.
6. Check the printed metrics and generated files in `data/saved/`.

**Important:** Run Jupyter from the repository root so relative paths such as `data/aclImdb/` resolve correctly. Full preprocessing of 50,000 reviews may take considerable time.

## Limitations

TF-IDF with a unigram-based classifier does not explicitly model word order, sarcasm, or broader linguistic context. Results are specific to the IMDb review dataset and experimental design described above.

## Dataset Reference

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142-150. https://aclanthology.org/P11-1015/

