import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
import string


def scrape_and_save_content(id, url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:

        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the title of the webpage
        title = soup.title.string

        # Extract main text body
        text_body = ""
        main_content = soup.find('body')
        if main_content:
            text_body = ' '.join([p.get_text() for p in main_content.find_all('p')])

        # Save the content to a text file with the name as the ID
        with open(fr"C:\Users\prabhu.patrot\Blackcoffer\Extracted text files\{id}.txt", "w", encoding="utf-8") as file:
            file.write(f"Title: {title}\n\n")
            file.write(text_body)

        print(f"Content saved for ID: {id}")
    else:
        print(f"Failed to fetch content for ID: {id}")


# Read the Excel file into a DataFrame
df = pd.read_excel(
    r"C:\Users\prabhu.patrot\Blackcoffer\20211030 Test Assignment-20240511T052222Z-001\20211030 Test Assignment\Input.xlsx")

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['URL_ID'])
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    id = row['URL_ID']
    url = row['URL']

    # Scrape and save content for each URL
    # scrape_and_save_content(id, url)

# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')


# Function to extract text from files
def extract_text_from_file(file_path):
    print(f'In extract_text_from_file')
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Function to perform sentiment analysis
def calculate_sentiment_scores(text, positive_dict, negative_dict):
    print(f'In calculate_sentiment_scores')
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Initialize scores
    positive_score = 0
    negative_score = 0

    # Iterate over tokens to calculate scores
    for token in tokens:
        if token in positive_dict:
            positive_score += 1
        elif token in negative_dict:
            negative_score += 1

    # Calculate polarity score
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)

    # Calculate subjectivity score
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)

    return positive_score, negative_score, polarity_score, subjectivity_score


def calculate_gunning_fox_index(text):
    print(f'In calculate_gunning_fox_index')
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())  # Convert to lowercase for consistency

    # Calculate average sentence length
    average_sentence_length = len(words) / len(sentences)

    # Identify complex words (words with more than 2 syllables)
    syllable_count = lambda word: max(1, sum(map(lambda word: 1 if word[-1] in ['.', ',', '?', '!'] else 0, word)))
    complex_words = [word for word in words if syllable_count(word) > 2]

    # Calculate percentage of complex words
    percentage_complex_words = len(complex_words) / len(words) * 100

    # Calculate Fog Index
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)

    return average_sentence_length, percentage_complex_words, fog_index


def calculate_average_words_per_sentence(text):
    print(f'In calculate_average_words_per_sentence')
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())  # Convert to lowercase for consistency

    # Calculate average number of words per sentence
    average_words_per_sentence = len(words) / len(sentences)

    return average_words_per_sentence


def count_cleaned_words(text):
    print(f'In count_cleaned_words')
    # Tokenize words
    words = word_tokenize(text.lower())  # Convert to lowercase for consistency

    # Remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Count cleaned words
    cleaned_word_count = len(words)

    return cleaned_word_count


def count_syllables(word):
    print(f'In count_syllables')
    # Define exceptions where "es" and "ed" are not counted as syllables
    exceptions = ['es', 'ed']
    # Count vowels in the word
    vowels = 'aeiouy'
    num_syllables = 0
    prev_char_was_vowel = False
    for char in word:
        if char.lower() in vowels:
            # Increment syllable count if current char is a vowel and previous char was not
            if not prev_char_was_vowel:
                num_syllables += 1
            prev_char_was_vowel = True
        else:
            # Check for exceptions like "es" and "ed"
            if char.lower() + (
                    word[word.index(char) + 1] if word.index(char) + 1 < len(word) else '') not in exceptions:
                prev_char_was_vowel = False
    # Handle special case where ending with "e" and not "es" or "ed"
    if word.endswith('e') and word[-2:] not in exceptions:
        num_syllables -= 1
    # Ensure minimum syllable count of 1
    return max(num_syllables, 1)


def count_syllables_in_text(text):
    print(f'In count_syllables_in_text')
    # Tokenize text into words
    words = text.split()
    # Count syllables for each word
    syllable_counts = [count_syllables(word) for word in words]
    return syllable_counts


def count_complex_words(text):
    print(f'In count_complex_words')
    # Tokenize words
    words = word_tokenize(text.lower())  # Convert to lowercase for consistency

    # Define a function to count syllables in a word
    syllable_count = lambda word: max(1, sum(map(lambda word: 1 if word[-1] in ['.', ',', '?', '!'] else 0, word)))

    # Count complex words (words with more than two syllables)
    complex_words = [word for word in words if syllable_count(word) > 2]

    # Calculate the number of complex words
    complex_word_count = len(complex_words)
    return complex_word_count


def count_personal_pronouns(text):
    print(f'In count_personal_pronouns')
    # Define the list of personal pronouns
    personal_pronouns = ['I', 'we', 'my', 'ours', 'us']
    # Define the regex pattern to match the personal pronouns
    pattern = r'\b(?:' + '|'.join(personal_pronouns) + r')\b'
    # Compile the regex pattern
    regex = re.compile(pattern, flags=re.IGNORECASE)
    # Find all matches of personal pronouns in the text
    matches = regex.findall(text)
    # Exclude occurrences of 'US' as a country name
    matches = [match for match in matches if match.lower() != 'us']
    # Count the occurrences of personal pronouns
    pronoun_count = len(matches)
    return pronoun_count


def calculate_average_word_length(text):
    print(f'In calculate_average_word_length')
    # Tokenize words
    words = text.split()
    # Calculate total number of characters in all words
    total_characters = sum(len(word) for word in words)
    # Calculate total number of words
    total_words = len(words)
    # Calculate average word length
    if total_words > 0:
        average_word_length = total_characters / total_words
    else:
        average_word_length = 0
    return average_word_length


# Function to load stop words from file
def load_stop_words(stop_words_folder):
    print(f'In load_stop_words')
    stop_words = set()
    # Iterate over files in the folder
    for filename in os.listdir(stop_words_folder):
        if filename.endswith(".txt"):  # Assuming all stop words files have .txt extension
            file_path = os.path.join(stop_words_folder, filename)
            # Read stop words from the file and add them to the set
            with open(file_path, 'r', encoding='latin-1') as file:
                stop_words.update(file.read().splitlines())
    return stop_words


# Function to load dictionary from file
def load_dictionary(dictionary_file):
    print(f'In load_dictionary')
    with open(dictionary_file, 'r', encoding='latin-1') as file:
        dictionary = set(file.read().splitlines())
    return dictionary


# Load stop words
stop_words_folder = r"C:\Users\prabhu.patrot\Blackcoffer\20211030 Test Assignment-20240511T052222Z-001\20211030 Test Assignment\StopWords"
stop_words = load_stop_words(stop_words_folder)
print(len(stop_words))

# Load positive and negative dictionaries
positive_dict_file = r"C:\Users\prabhu.patrot\Blackcoffer\20211030 Test Assignment-20240511T052222Z-001\20211030 Test Assignment\MasterDictionary\positive-words.txt"
negative_dict_file = r"C:\Users\prabhu.patrot\Blackcoffer\20211030 Test Assignment-20240511T052222Z-001\20211030 Test Assignment\MasterDictionary\negative-words.txt"
positive_dict = load_dictionary(positive_dict_file) - stop_words
negative_dict = load_dictionary(negative_dict_file) - stop_words

# Define folder path containing files
folder_path = r"C:\Users\prabhu.patrot\Blackcoffer\Extracted text files"

# Create an empty DataFrame to store results
results_df_1 = pd.DataFrame(
    columns=['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
             'AVG SENTENCE LENGTH',
             'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
             'WORD COUNT',
             'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # Extract text from file
        file_path = os.path.join(folder_path, filename)
        text = extract_text_from_file(file_path)

        # Clean the text by removing stop words
        cleaned_text = ' '.join([word for word in text.lower().split() if word not in stop_words])

        # Calculate sentiment scores
        positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentiment_scores(cleaned_text,
                                                                                                        positive_dict,
                                                                                                        negative_dict)

        # Calculate Gunning Fox index
        average_sentence_length, percentage_complex_words, fog_index = calculate_gunning_fox_index(text)

        # Calculate average number of words per sentence
        average_words_per_sentence = calculate_average_words_per_sentence(text)

        # Count complex words
        complex_word_count = count_complex_words(text)

        # Count cleaned words
        cleaned_word_count = count_cleaned_words(text)

        # count syllables
        syllable_counts = count_syllables_in_text(text)

        # Count personal pronouns
        pronoun_count = count_personal_pronouns(text)

        # Count average word length
        average_word_length = calculate_average_word_length(text)

        # Add results to DataFrame
        new_row = {
            'URL_ID': filename, 'POSITIVE SCORE': positive_score, 'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': polarity_score, 'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': average_sentence_length,
            'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words, 'FOG INDEX': fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': average_words_per_sentence,
            'COMPLEX WORD COUNT': complex_word_count, 'WORD COUNT': cleaned_word_count,
            'SYLLABLE PER WORD': syllable_counts,
            'PERSONAL PRONOUNS': pronoun_count, 'AVG WORD LENGTH': average_word_length
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

# Save results to Excel file
results_df.to_excel(r"C:\Users\prabhu.patrot\Blackcoffer\text_analysis_results.xlsx", index=False)
