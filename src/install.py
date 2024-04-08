import os
import urllib.request
import zipfile
import tarfile

# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)


# word2vecの日本語学習済みモデル（東北大学 乾・岡崎研究室）をダウンロード。時間が15分ほどかかります

url = "http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2"
save_path = "./data/20170201.tar.bz2"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

# './data/20170201.tar.bz2'の解凍　5分ほどかかります

# tarファイルを読み込み
tar = tarfile.open('./data/20170201.tar.bz2', 'r|bz2')
tar.extractall('./data/')  # 解凍
tar.close()  # ファイルをクローズ

# フォルダ「data」内にフォルダ「entity_vector」というものができ、その中に「entity_vector.model.bin」というファイルができています。

# fastTextの公式の英語学習済みモデル（650MB）をダウンロード。時間が5分ほどかかります
url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
save_path = "./data/wiki-news-300d-1M.vec.zip"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

# フォルダ「data」内の「/wiki-news-300d-1M.vec.zip」を解凍する

zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
zip.extractall("./data/")  # ZIPを解凍
zip.close()  # ZIPファイルをクローズ

# フォルダ「data」内にフォルダ「wiki-news-300d-1M.vec」というものができます。

# IMDbデータセットをダウンロード。30秒ほどでダウンロードできます

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)


# # './data/aclImdb_v1.tar.gz'の解凍　1分ほどかかります

# # tarファイルを読み込み
# tar = tarfile.open('./data/aclImdb_v1.tar.gz')
# tar.extractall('./data/')  # 解凍
# tar.close()  # ファイルをクローズ

# # フォルダ「data」内にフォルダ「aclImdb」

# # フォルダ「data」内の「vector_neologd.zip」を解凍する

# zip = zipfile.ZipFile("./data/vector_neologd.zip")
# zip.extractall("./data/vector_neologd/")  # ZIPを解凍
# zip.close()  # ZIPファイルをクローズ