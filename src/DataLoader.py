# tsv形式のファイルにします
import glob
import os
import io
import string


# 訓練データのtsvファイルを作成します

f = open('./data/IMDb_train.tsv', 'w')

path = './data/aclImdb/train/pos/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)

path = './data/aclImdb/train/neg/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)

f.close()

# テストデータの作成

f = open('./data/IMDb_test.tsv', 'w')

path = './data/aclImdb/test/pos/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)


path = './data/aclImdb/test/neg/'

for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)

f.close()






"****************************"
# 2. 前処理と単語分割の関数を定義
"****************************"





import string
import re

# 以下の記号はスペースに置き換えます（カンマ、ピリオドを除く）。
# punctuationとは日本語で句点という意味です
print("区切り文字：", string.punctuation)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# 前処理


def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text

# 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）


def tokenizer_punctuation(text):
    return text.strip().split()


# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


# 動作を確認します
print(tokenizer_with_preprocessing('I like cats.'))





"****************************"
# DataLoaderの作成
"****************************"





# データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
import torchtext


# 文章とラベルの両方に用意します
max_length = 256
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

# 引数の意味は次の通り
# init_token：全部の文章で、文頭に入れておく単語
# eos_token：全部の文章で、文末に入れておく単語

" メモ "
# eos = En of Sentence
# cls = Class (bos = Begin of Sentence が使われることが多いが今回はクラス分類なので、cls)





# フォルダ「data」から各tsvファイルを読み込みます
train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='./data/', train='IMDb_train.tsv',
    test='IMDb_test.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)])

# 動作確認
print('訓練および検証のデータ数', len(train_val_ds))
print('1つ目の訓練および検証のデータ', vars(train_val_ds[0]))


import random
# torchtext.data.Datasetのsplit関数で訓練データとvalidationデータを分ける

train_ds, val_ds = train_val_ds.split(
    split_ratio=0.8, random_state=random.seed(1234))

# 動作確認
print('訓練データの数', len(train_ds))
print('検証データの数', len(val_ds))
print('1つ目の訓練データ', vars(train_ds[0]))






"****************************"
# ボキャブラリーを作成
"****************************"





# torchtextで単語ベクトルとして英語学習済みモデルを読み込みます

from torchtext.vocab import Vectors

english_fasttext_vectors = Vectors(name='data/wiki-news-300d-1M.vec')


# 単語ベクトルの中身を確認します
print("1単語を表現する次元数：", english_fasttext_vectors.dim)
print("単語数：", len(english_fasttext_vectors.itos))


# ベクトル化したバージョンのボキャブラリーを作成します
TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

# ボキャブラリーのベクトルを確認します
print(TEXT.vocab.vectors.shape)  # 17916個の単語が300次元のベクトルで表現されている
# TEXT.vocab.vectors
print(TEXT.vocab.vectors)

# ボキャブラリーの単語の順番を確認します
# TEXT.vocab.stoi
print(TEXT.vocab.stoi)


# DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
train_dl = torchtext.data.Iterator(train_ds, batch_size=24, train=True)

val_dl = torchtext.data.Iterator(
    val_ds, batch_size=24, train=False, sort=False)

test_dl = torchtext.data.Iterator(
    test_ds, batch_size=24, train=False, sort=False)


# 動作確認 検証データのデータセットで確認
batch = next(iter(val_dl))
print(batch.Text)
print(batch.Label)
