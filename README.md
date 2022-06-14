# auto-img-tag

物体検出の学習データを自動生成します。

Google Colabで学習データの自動生成からYOLOv5による物体検出までできるのでお試しください。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vy5CFLmAdbjK-6n-qyJUyOlUtFVIotZ5?usp=sharing)

# 目次

* [インストールの方法](#インストールの方法)
* [学習データの作成の手順](#学習データの作成の手順)
    * [1. カメラ アプリで動画ファイルを作成する。](#1-カメラ-アプリで動画ファイルを作成する)
    * [2. クラス別に動画ファイルをフォルダに入れる。](#2-クラス別に動画ファイルをフォルダに入れる)
    * [3. 背景画像ファイルを準備する。](#3-背景画像ファイルを準備する)
    * [4. 学習データ作成アプリを実行する。](#4-学習データ作成アプリを実行する)
* [GUIアプリ](#guiアプリ)

<a id="install"></a>
## インストールの方法

pipをアップグレードしておきます。
```bash
python -m pip install --upgrade pip
```

必要なパッケージをインストールします。
```bash
pip install -U numpy opencv-python Pillow tqdm pysimplegui albumentations
```

適当なフォルダでソースをクローンします。
```bash
git clone https://github.com/teatime77/auto-img-tag.git
```


<a id="tejun"></a>
## 学習データの作成の手順

1. [カメラ アプリで動画ファイルを作成する。](#tejun-1)
2. [クラス別に動画ファイルをフォルダに入れる。](#tejun-2)
3. [背景画像ファイルを準備する。](#tejun-3)
4. [学習データ作成アプリを実行する。](#tejun-4)


<a id="tejun-1"></a>
### 1. カメラ アプリで動画ファイルを作成する。

![camera](https://uroa.jp/auto-img-tag/img/camera.png)

#### 起動方法
camera.pyを実行します。

```bash
python camera.py
```

#### 操作方法

- 左の画面は明度で二値化する場合です。<br/>
  **明度の閾値** を変えると二値画像の白の領域が変化します。<br/>
  物体が背景から分離されるように調整します。

- 右の画面は色相で二値化する場合です。

  1. 物体を置かずに **色相を指定** ボタンをクリックして、画像の中心の色を背景色として登録します。

  2. 物体を置いてから **色相の範囲** を変えると二値画像の白の領域が変化します。<br/>
  物体が背景から分離されるように調整します。

- **動画撮影** ボタンをクリックすると録画が始まり、 **停止** ボタンで録画を終了します。<br/>
  動画ファイルはカレントフォルダの直下の **capture** フォルダに保存されます。

- **写真撮影** ボタンをクリックすると原画の静止画が **capture** フォルダに保存されます。

---

<a id="tejun-2"></a>
### 2. クラス別に動画ファイルをフォルダに入れる。

以下は動画ファイルのフォルダ構成の例です。

![foler](https://uroa.jp/auto-img-tag/img/folder.png)

vegetable-videoフォルダの下に onion, potato, tomato のフォルダがあり、それぞれのフォルダの中に動画ファイルが入っています。

この例ではonionには３個の動画ファイル、potatoとtomatoには２個の動画ファイルが入っています。

学習データを作成するとき、onion, potato, tomatoがクラス名(カテゴリー名)になります。
動画ファイルのファイル名は何でも構いません。

ただしOpenCVは日本語のファイル名に対応していないので、フォルダ名やファイル名には日本語を含めないでください。

---

<a id="tejun-3"></a>
### 3. 背景画像ファイルを準備する。

背景画像ファイルとして何でも良いのですが、COCOデータセットの **2017 Val images** を使っています。

[https://cocodataset.org/](https://cocodataset.org/)

COCOの2017 Val imagesは5000枚あります。

---

<a id="tejun-4"></a>
### 4. 学習データ作成アプリを実行する。

main.pyを実行します。

```bash
python main.py -i 動画ファイルのフォルダ -bg 背景画像のフォルダ -o 出力先のフォルダ -dtsz １クラス当たりの学習データの数 -imsz 出力画像のサイズ
```

実行すると出力先のフォルダに **train.json** と **img** フォルダができます。

<pre>
┳ train.json
┗ img
    ┣ 画像ファイル
    ⋮
    ┗ 画像ファイル
</pre>

train.jsonはCOCO形式でアノテーションの情報が書かれています。

imgフォルダの下に学習用の画像ファイルが作成されます。
<br/>

#### main.pyのコマンドライン引数の一覧
<dl>
  <dt>-i</dt>
  <dd>動画ファイルのフォルダ</dd>
  <dt>-bg</dt>
  <dd>背景画像ファイルのフォルダ</dd>
  <dt>-o</dt>
  <dd>学習データの出力先のフォルダ</dd>
  <dt>-dtsz</dt>
  <dd>1クラスあたりの学習データの数。デフォルトは1000。</dd>
  <dt>-imsz</dt>
  <dd>出力画像のサイズ。 デフォルトは720。</dd>
  <dt>-v</dt>
  <dd>明度の閾値。デフォルトは250。</dd>
</dl>


---

### GUIアプリ

![foler](https://uroa.jp/auto-img-tag/img/gui.png)

GUIアプリはデータ拡張などのデバッグに使っています。

起動の引数はmain.pyと同じです。

```bash
python gui.py -i 動画ファイルのフォルダ -bg 背景画像のフォルダ -o 出力先のフォルダ
```
