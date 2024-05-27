# Sentiment-analysis-and-text-summarization

## Overview
This project is designed to perform sentiment analysis and text analysis on a collection of web articles. The process involves scraping web content, processing the text, calculating various text metrics, and performing sentiment analysis. The results are saved into an Excel file for further analysis.

## Prerequisites
Ensure you have the following Python libraries installed:

1. requests
2. beautifulsoup4
3. pandas
4. nltk

You can install these libraries using pip:
```bash
pip install requests beautifulsoup4 pandas nltk
```

Additionally, make sure you have the NLTK resources for tokenization and stop words in python script:
```python
nltk.download('punkt')
nltk.download('stopwords')
```

## File Structure
1. input.xlsx: An Excel file containing URLs to scrape.
2. stopwords: A folder containing stop words files.
3. positive-words.txt: A file containing positive words for sentiment analysis.
4. negative-words.txt: A file containing negative words for sentiment analysis.
5. Extracted text files: A folder where the scraped web content will be saved.
6. text_analysis_results.xlsx: The output Excel file with the analysis results.

## Code Explanation
## Step 1: Scraping Web Content
The function scrape_and_save_content(id, url) scrapes the content of the given URL and saves it to a text file named with the corresponding ID.

## Step 2: Loading Input Data
The code reads the URLs from input.xlsx and iterates over each URL to scrape and save its content.

## Step 3: Downloading NLTK Resources
The required NLTK resources for tokenization and stop words are downloaded.

## Step 4: Text Processing Functions
Several functions are defined to perform text processing and analysis:

1. extract_text_from_file(file_path): Extracts text from a given file.
2. calculate_sentiment_scores(text, positive_dict, negative_dict): Calculates sentiment scores.
3. calculate_gunning_fox_index(text): Calculates the Gunning Fog Index.
4. calculate_average_words_per_sentence(text): Calculates the average number of words per sentence.
5. count_cleaned_words(text): Counts the number of cleaned words (removing stop words and punctuation).
6. count_syllables(word): Counts the syllables in a word.
7. count_syllables_in_text(text): Counts the syllables in the entire text.
8. count_complex_words(text): Counts the number of complex words.
9. count_personal_pronouns(text): Counts the number of personal pronouns.
10. calculate_average_word_length(text): Calculates the average word length.

## Step 5: Loading Stop Words and Dictionaries
Stop words and sentiment dictionaries (positive and negative words) are loaded from files.

## Step 6: Performing Analysis
The text files are processed to calculate various metrics, including sentiment scores, readability scores, word counts, and syllable counts. The results are saved into an Excel file text_analysis_results.xlsx.

## How to Use
1. Ensure your file structure matches the expected structure mentioned above.
2. Run the script. The script will:
   1. Read URLs from input.xlsx.
   2. Scrape and save web content into the Extracted text files folder.
   3. Load stop words and sentiment dictionaries.
   4. Process each saved text file to calculate various metrics.
   5. Save the analysis results into text_analysis_results.xlsx.