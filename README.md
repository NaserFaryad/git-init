# Semantic Segmentation Library

This is a generic Semantic Segmentation library. Multi-class classification for each pixel can be done with minimal preparation.

## Building an environment with Google Colab
Most of the time, this script is executed by Google Colab, so I made it easy to build the environment with Colab.
You no longer have to go through the locals.
`train_inference.ipynb` Added the part to clone from git and download additional resources.

1. Open it with Colab from the button on the right and copy it to the drive.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kubotaissei/misc/blob/master/train_inference.ipynb)
2. `train_inference.ipynb`From top to bottom．

## Environment construction (when constructing a virtual environment)
We recommend 3.6.9 for Python and 2.3.x for Tensorflow. It may not work with Python 3.8 series.

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
During operation verification
```

[DL link for additional resources](https://www.dropbox.com/sh/i2r8t74riiijw5p/AADz1guLg0-x__EGM5_t4nUJa?dl=0)

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
