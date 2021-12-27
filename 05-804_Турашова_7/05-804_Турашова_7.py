"""
7. Наивный байесовский классификатор текстов на основе открытых данных

Дано:
- Наивный байесовский классификатор текстов

Требуется:
- Выбрать тематику текстов для классификации (например, научить модель отличать спортивные тексты от всех остальных)
- Выбрать источники и методы сбора открытых данных (например, тематические сайты и библиотеки парсинга текстов, или
социальная сеть ВКонтакте с ее API)
- собрать достаточную для классификации подборку текстов (от сотни; чем больше и разнообразнее, тем лучше)
- выбрать методы и библиотеки предобработки текстов
- произвести обучение модели и выгрузить словари во внешний pkl файл
- проверить результаты классификации на тестовых текстах
- подготовить следующие файлы для отправки:
1) pkl файл со словарями
2) код программы (который можно протестировать с произвольными тестовыми текстами)
3) отчет в произвольной форме о проделанной работе doc/ipynb (выбор тематики, источников, методов обработки текстов,
объем обучающей выборки...)
- запаковать все файлы в один архив и назвать файл по шаблону на латинице без пробелов: Группа_Фамилия_НомерРаботы
(например, 11-000_Ivanov_7)
"""

from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
import numpy as np
import pickle
import random
import requests


# Один раз скачать:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


word_net_lemmatizer = WordNetLemmatizer()
regexp_tokenizer = RegexpTokenizer(r'\w+')


def get_content(html, num):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('span', class_='card-mini__title')
    news = []
    for i, item in enumerate(items):
        news.append([item.text, num])
    print(news)
    return news


URL_media = 'https://lenta.ru/rubrics/media/'
news_media = get_content(requests.get(URL_media).text, 0)

URL_travel = 'https://lenta.ru/rubrics/travel/'
news_travel = get_content(requests.get(URL_travel).text, 1)


# Возьмём 90% на обучение и 10% на тест:
train_test_ratio = 0.9

random.shuffle(news_media)
media_train = news_media[:round(len(news_media)*train_test_ratio)]
media_test = news_media[round(len(news_media)*train_test_ratio):]
print(f'Новостей, относящихся к media:\n{len(media_train)} обучающих, {len(media_test)} тестовых')

random.shuffle(news_travel)
travel_train = news_travel[:round(len(news_travel)*train_test_ratio)]
travel_test = news_travel[round(len(news_travel)*train_test_ratio):]
print(f'Новостей, относящихся к travel:\n{len(travel_train)} обучающих, {len(travel_test)} тестовых')

train = media_train + travel_train
test = media_test + travel_test


def prepare_data(dataset):
    result = dataset.copy()
    for i in range(len(dataset)):
        tokens = regexp_tokenizer.tokenize(dataset[i][0])
        processed = []
        for token in tokens:
            if token.lower() not in nltk.corpus.stopwords.words('russian'):
                processed += [word_net_lemmatizer.lemmatize(token.lower())]
        result[i][0] = processed
    return result


prepared_train = prepare_data(train)
print('Подготовленные данные для обучения:', prepared_train)

prepared_test = prepare_data(test)
print('Подготовленные данные для теста:', prepared_test)


class NaiveBayesClf:
    def __init__(self, alpha=0.01):
        self.classes = {}
        self.freq = {}
        self.total_in_class = {}
        self.total = set()
        self.alpha = alpha

    def fit(self, dataset):
        self.dataset = dataset
        for features, label in self.dataset:
            if label not in self.classes:
                self.classes[label] = 0
                self.total_in_class[label] = 0
            self.classes[label] += 1
            for feature in features:
                if (feature, label) not in self.freq:
                    self.freq[(feature, label)] = 0
                self.freq[(feature, label)] += 1
                self.total_in_class[label] += 1
                self.total.add(feature)

        for feature, label in self.freq:
            self.freq[(feature, label)] = (self.alpha + self.freq[(feature, label)]) / (self.alpha * len(self.total)
                                                                                        + self.total_in_class[label])
        for cls in self.classes:
            self.classes[cls] /= len(self.dataset)
        return self

    def predict(self, features):
        return max(self.classes.keys(),
                   key=lambda cls: np.log10(self.classes[cls]) + sum(np.log10(self.freq.get((feature, cls),
                                            self.alpha / (self.alpha * len(self.total) + self.total_in_class[cls]))) \
                                       for feature in features))

    def save(self, path: str):
        dump = {
            'classes': self.classes,
            'freq': self.freq,
            'total_in_class': self.total_in_class,
            'total': self.total,
            'alpha': self.alpha
        }
        with open(path, 'wb') as f:
            pickle.dump(dump, f)
        print("Model has been saved by following path: {}".format(path))

    def load(self, path: str):

        with open(path, 'rb') as f:
            dump = pickle.load(f)
        self.classes = dump['classes']
        self.freq = dump['freq']
        self.total_in_class = dump['total_in_class']
        self.total = dump['total']
        self.alpha = dump['alpha']
        return self


model = NaiveBayesClf(0.1).fit(prepared_train)
model.save("model.pkl")


all_pred = 0
true_pred = 0
for i in range(len(prepared_train)):
    if prepared_train[i][1] == model.predict(prepared_train[i][0]):
        true_pred += 1
    all_pred += 1
print("Точность на обучающей выборке: {:.2f}%".format(true_pred/all_pred))


all_pred = 0
true_pred = 0
for i in range(len(prepared_test)):
    if prepared_test[i][1] == model.predict(prepared_test[i][0]):
        true_pred += 1
    all_pred += 1
print("Точность на тестовой выборке: {:.2f}%".format(true_pred/all_pred))


# Можно вписать любую новость и вручную проверить принадлежность к группам:
# tokens = regexp_tokenizer.tokenize("Собчак прокомментировала награду дочери Рамзана Кадырова")
tokens = regexp_tokenizer.tokenize("Авиапассажир подал в суд после травмы от падения чемодана на голову и продешевил")
processed = []
for token in tokens:
    if token.lower() not in nltk.corpus.stopwords.words('russian'):
        processed += [word_net_lemmatizer.lemmatize(token.lower())]
print("Относится к media" if model.predict(processed) == 0 else "Относится к travel")
