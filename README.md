# PyTorch-Autoencoder using MNIST

Powered by [satolab](https://qiita.com/satolab)

## Overview

時系列モデルであるGRUと，encoder-decoderモデルを組み合わせた，動画再構成モデルです．
ここではこのモデルを，GRU-AEと呼びます．



## Model

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/c5ddc1c8-f82d-b6ff-8cca-b9cc7a2787f8.png" width="400×200">

## Results
- 10 epochs(input,output,difference)


## Usage
- main.pyで学習．save dirにサンプルが保存されます．
Learn with main.py.

The sample is saved in save dir.

## References
差分画像の計算と表示部分
http://cedro3.com/ai/keras-autoencoder-anomaly/
