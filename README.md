# correct aspect ratio
## データセット作成手順

### fuelを準備する
まずfuelをインストールする:

```
pip install fuel==0.2.0
```

次にfuelのルートディレクトリを設定する。例えば `E:\Dataset\fuel` をfuelのルートディレクトリとして指定するなら、ホームディレクトリに `.fuelrc` ファイルを作成し、以下の内容で保存する:

```
data_path: E:\Dataset\fuel
```

### データセットをダウンロード
-  [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) から[Images](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) (757MB) をダウンロードする

### データセットを展開
-  ターミナルを起動
-  Imagesを保存したディレクトリに移動  
`$ cd data_location`
-  以下のコマンドを実行  
`$ tar xvf images.tar`  
-  データセットが展開される

### HDF5ファイルを作成
- create_dogdataset.py の以下のパラメータを自身の設定に変更する
    -  data_location : データセットを展開したディレクトリのルートパス
    -  output_location : HDF5ファイルを保存するディレクトリのルートパス
    -  output_size : hdf5ファイルに保存する画像サイズ(例：output_size=500の場合、保存される画像サイズは500 * 500)
-  create_dogdataset.pyを実行
