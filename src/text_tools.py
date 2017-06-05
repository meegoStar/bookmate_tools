import json
import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

stopwords = set()
books_dict = {}
corpus = []
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def set_jieba():
    traditional_chinese_dict_path = '../data/dict.txt.big'
    jieba.set_dictionary(traditional_chinese_dict_path)

def set_stopwords():
    stopwords_path = '../data/stop_words.txt'

    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip('\n'))

    stopwords.add('\n')
    stopwords.add(' ')
    return stopwords

def load_books():
    library_books_path = '../data/books_list.json'
    eslite_books_path = '../data/誠品書籍資料.json'
    
    with open(library_books_path) as f:
        library_books = json.load(f)

    with open(eslite_books_path) as f:
        eslite_books = json.load(f)

    return library_books, eslite_books

def build_books_dict():
    library_books, eslite_books = load_books()

    for book in library_books:
        title = book['book name']
        num = book['book number']

        books_dict[title] = {}
        books_dict[title]['name'] = title
        books_dict[title]['num'] = num
        books_dict[title]['abstract'] = ''
        books_dict[title]['publish'] = ''
        books_dict[title]['author'] = ''
        books_dict[title]['author_intro'] = ''
        books_dict[title]['catelog'] = ''
        books_dict[title]['category'] = ''

    for index, book in enumerate(eslite_books):
        title = book['bookname']
        abstract = book['abstract']
        catelog = book['catelog']
        category = book['category']

        if title in books_dict.keys():
            books_dict[title]['abstract'] += ' ' + abstract
            books_dict[title]['catelog'] += ' ' + catelog
            books_dict[title]['category'] += ' ' + category

    print('Loaded books:', len(books_dict.keys()))

def segment_text(text):
    words = list(jieba.cut(text, cut_all=False))
    for word in words:
        if word in stopwords or word.isdigit():
            words.remove(word)

    joined = [' '.join(words)]
    return joined

def build_corpus():
    print('Building corpus...')
    global corpus

    length = len(books_dict.keys())
    for count, title in enumerate(books_dict.keys()):
        abstract = books_dict[title]['abstract']
        catelog = books_dict[title]['catelog']
        category = books_dict[title]['category']
        # segment all the info of each book
        title_segmented = segment_text(title)
        abstract_segmented = segment_text(abstract)
        catelog_segmented = segment_text(catelog)
        category_segmented = segment_text(category)
        info_segmented = title_segmented + abstract_segmented + catelog_segmented + category_segmented
        info_concatenated = ' '.join(list(filter(None, info_segmented)))
        corpus.append(info_concatenated)
        # print progress
        print('%.2f' % float(count / length * 100), '%', end = '\r')

    print('Corpus builded.')

def train_tfidf():
    print('Training TF-IDF...')
    X_train_counts = count_vectorizer.fit_transform(corpus)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print('TF-IDF trained.')

def initialize_text_tools():
    set_jieba()
    set_stopwords()
    build_books_dict()
    build_corpus()
    train_tfidf()
    print('Text-tools initialized.')

def test_tfidf(text_segmented):
    X_new_counts = count_vectorizer.transform(text_segmented)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    return X_new_tfidf

def display_tfidf(tfidf):
    word = count_vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for i in range(len(weight)):
        print("-------這裡輸出第", i, "類文本的詞語 tf-idf 權重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])

if __name__ == '__main__':
    # Initialize all needed for text-tools
    initialize_text_tools()
    # Test TF-IDF
    test_text_segmented = ['曹操 三國 演義']
    tfidf = test_tfidf(test_text_segmented)
    print(tfidf)
