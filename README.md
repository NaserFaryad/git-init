# Semantic Segmentation Library

これは汎用的なSemantic Segmentation用ライブラリです。ピクセルごとの多クラス分類を、最低限の準備で行えます。

## Google Colabでの環境構築
大抵このスクリプトはGoogle Colabで実行することが多そうなので，Colabでの環境構築を簡単にしました．
ローカルを経由しなくてもよくなりました．
`train_inference.ipynb`内に，gitからcloneして追加リソースをダウンロードする部分を追加しました．

1. 右のボタンからColabで開き，ドライブにコピーする．
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kubotaissei/misc/blob/master/train_inference.ipynb)
2. `train_inference.ipynb`を上から順に実行する．

## 環境構築（仮想環境を構築する場合）
Pythonは3.6.9、Tensorflowは2.3.xを推奨します。Python3.8系だと動かないかもしれません。

MacOS:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements/mac_0828.txt
```

Windows:
```
python -m venv venv
venv/Scripts/activate.bat
pip install -r requirements/mac_0828.txt
```

Ubuntu:
```
動作検証中
```

[追加リソースのDLリンク](https://www.dropbox.com/sh/i2r8t74riiijw5p/AADz1guLg0-x__EGM5_t4nUJa?dl=0)

Download the two zip files from the link above. 
- `dataset_and_inference_data.zip`: Extract it and place the three folders inside in the project root.
- `pretrained_weights.zip`: Extract and place the folder itself in `src / models /`.

Check if the directory structure is as follows.
- dataset
- inference_target
- inference_result
- requirements
- src
    - utils
    - models
        - pretrained_weights
- venv
- weights: Created automatically during learning


## Algorithms currently available

- UNet
- Deeplab V3 Plus


## How to learn

1. Prepare the dataset

`dataset/`And prepare the directory of the data set to be used this time (`sample /` this time) in it.
Create the following four directories in it and store images in each.
- `dataset/sample/train/image`: 訓練用画像
- `dataset/sample/train/label`: 訓練用ラベル
- `dataset/sample/test/image`: 評価用画像
- `dataset/sample/test/label`: 評価用ラベル

2. Specify the dataset and label information to use

`src/train.py`の`if __name__ == "__main__":`among,

``` python
dataset_name = "sample"
```
will do.

In addition, label information is defined by the OrderedDict type. The key should be the class name and the value should be the RGB value in the label image of that class.
``` python
label_info = collections.OrderedDict()
label_info["background"] = (0, 0, 0)
label_info["trash"] = (128, 0, 0)
label_info["rice"] = (0, 128, 0)
```


3. Specify the model to use

At this time, `UNet` or` DeepLab V3 Plus` is available. In `src / train.py`, remove the` # `of the model you want to use.At this time, `UNet` or` DeepLab V3 Plus` is available. In `src / train.py`, remove the` # `of the model you want to use.

``` python
# model = UNet(input_shape=img_shape, classes=len(label_info))
model = Deeplabv3(input_shape=img_shape, classes=len(label_info), backbone="xception", activation='softmax')
```

4. Change the learning parameters to the optimum
``` python
run(
    img_shape=(512, 512, 3),  # Of the training image shape (height, width, channel)
    steps_per_epoch=200,
    epochs=300,
    validation_steps=8,
    batch_size=1,
    lr=1e-3,  # Optimizer Learning rate
    weights_of_crossentropy=base_weights_of_crossentropy * np.array([1., 3., 1.]),
    dataset_path=dataset_path,
    label_info=label_info,
    description="sample_Adam_1e-4_reduce_lr_1e-6_5_2"  # Use to record what kind of setting you learned (free format)
)
```

- `img_shape`: Specifies the resolution of the image used for learning. The image in the dataset will be cropped to this size at random positions. Select `(256, 256)` or `(512, 512)`. It can be changed to other sizes, but it works normally only in multiples of 4 due to the setting at the time of inference.
- `weights_of_crossentropy`: This library uses weighted cross entropy instead of normal cross entropy.
This makes it possible to study efficiently even if there is a large bias in the frequency of appearance between classes.
By using the `calc_weights_of_crossentropy` function, you can automatically calculate the appropriate weight from the number of occurrences of each class. You can further weight it, such as `* np.array ([1., 3., 1.])`.
- `description`: It's difficult to remember what settings will work, so it's a good idea to write some identifiable content here in advance. The format is free.

5. Set the required Callback function

`CallbackForSegmentation`Provides the following features:

- Inference is made in the model after training of each epoch, and the result is saved under `weights / ~~~ / images /`. The file name includes the current number of epochs, class name, and IoU.
- Ground Truth Calculate mIoU by comparing with.
- This value is stored in `weights / ~~~ / log.csv` by the` CSV Logger` that comes standard with Keras. The `CSVLogger` must be placed after the` CallbackForSegmentation`.

6. Run
```
python stc/sample_train.py
```
A new folder will be created in `weights /` for each learning. Trained weights are also stored here.
`run()`Function`if __name__ == "__main__":`By repeating within, you can learn multiple settings in succession.

## Method of reasoning

1. Specify the model, the folder of the image to be inferred, and the save destination of the inference result.

``` python
# Pass / name settings
model_path = ""  # Enter with an absolute path
target_name = "genmai_masked"  # inference_target Folder name in
result_name = "genmai_masked_01"  # inference_result Folder name to create
```

ModelCheckpoint You can also specify the weight in the `log /` saved by.

2. Define the same label information that was used during learning.

``` python
# Definition of label information
label_info = collections.OrderedDict()
label_info["background"] = (0, 0, 0)
label_info["trash"] = (128, 0, 0)
label_info["rice"] = (0, 128, 0)
```

3. Run
``` python
python stc/sample_inference.py
```

