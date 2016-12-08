# correct aspect ratio
## データセット作成手順  

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
    -  output_size : 保存したいデータセットの画像サイズ
-  create_dogdataset.pyを実行
