# auto-img-tag

物体検出の学習データを自動生成します。


## インストールの方法

pipをアップグレードしておきます。
```bash
pip install --upgrade pip
```

必要なパッケージをインストールします。
```bash
pip install -U numpy opencv-python Pillow tqdm pysimplegui albumentations
```

適当なフォルダでソースをクローンします。
```bash
git clone https://github.com/teatime77/auto-img-tag.git
```


## 学習データの作成の手順

1. カメラ アプリで動画ファイルを作成する。
2. クラス別に動画ファイルをフォルダに入れる。
3. 背景画像ファイルを準備する。
4. 学習データ作成アプリを実行する。

以下でくわしく説明します。

### 1. カメラ アプリで動画ファイルを作成する。

![camera](https://user-images.githubusercontent.com/13596557/172041640-c332617a-13b4-49a9-9f56-aff9816372c3.jpg)

#### 起動方法
camera.pyを実行します。

```bash
python camera.py
```

#### 操作方法

* **明度の閾値** を調節して物体の輪郭が抽出されるようにします。
<br/>
* **動画撮影** ボタンをクリックすると録画が始まり、 **停止**ボタンで録画を終了します。
動画ファイルはカレントフォルダの直下の **capture** フォルダに保存されます。
<br/>
* **写真撮影** ボタンをクリックすると原画の静止画が **capture** フォルダに保存されます。
<br/>
### 2. クラス別に動画ファイルをフォルダに入れる。

以下は動画ファイルのフォルダ構成の例です。

![foler](https://uroa.jp/auto-img-tag/img/folder.png)

vegetable-videoフォルダの下に onion, potato, tomato のフォルダがあり、それぞれのフォルダの中に動画ファイルが入っています。

この例ではonionには３個の動画ファイル、potatoとtomatoには２個の動画ファイルが入っています。

学習データを作成するとき、onion, potato, tomatoがクラス名(カテゴリー名)になります。
動画ファイルのファイル名は何でも構いません。

ただしOpenCVは日本語のファイル名に対応していないので、フォルダ名やファイル名には日本語を含めないでください。

### 3. 背景画像ファイルを準備する。

背景画像ファイルとして何でも良いのですが、COCOデータセットの **2017 Val images** を使っています。
[https://cocodataset.org/](https://cocodataset.org/)

COCOの2017 Val imagesは5000枚あります。

### 4. 学習データ作成アプリを実行する。

```bash
python main.py -i 動画ファイルのフォルダ -bg 背景画像のフォルダ -o 出力先フォルダ
```

動画ファイルが **vegetable-video** フォルダにあり、背景画像が **val2017** フォルダにあり、**vegetable** フォルダに学習データを作成する場合は以下のようになります。

```bash
python main.py -i vegetable-video -bg val2017 -o vegetable
```






![gui](https://user-images.githubusercontent.com/13596557/172041725-233f1b25-42cf-4df8-af7c-96a65b3302f8.jpg)
