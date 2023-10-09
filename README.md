# Instructions
We will perform a video classification task using **audio-only features**. We will use **sklearn**, **pytorch**.

## Environment Settings
Suggest using [conda](https://docs.conda.io/en/latest/) to manage your packages. You can quickly check or install the required packages from `environment.yaml`.

If you use conda, you should easilly install the packages through:
```
conda env create -f environment.yaml
```

Major Depdencies we will use in this hw are: FFMPEG, Python, sklearn, pandas, pytorch, librosa

Install FFMPEG by:
```
$ apt install ffmpeg
```
Install python dependencies by (ignore if you start from .yaml): 
```
$ pip install scikit-learn==0.22 pandas tqdm librosa
```
If using conda, install pytorch by (ignore if you start from .yaml):
```
$ conda install pytorch torchvision torchaudio -c pytorch
```
Using gpu version of pytorch will significantly accelerate the feature extraction procedure. Please refer to [here](https://pytorch.org/get-started/locally/) for more detailed settings.

## Data and Labels
Please download the data from [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip) with wget. You could also download the data manually from [here](https://www.kaggle.com/competitions/cmu-11775-f23-hw1-audio-based-med/data). Then unzip it and put the videos under "$path_to_this_repo/videos", and labels under "$path_to_this_repo/labels". You can either directly download the data to this folder or in anywhere else then build a [soft link](https://linuxhint.com/create_symbolic_link_ubuntu/)

The `.zip` file should include the following:
1. `video/` folder with **8249 videos** in **MP4** format
2. `labels/` folder with two files:
    - `cls_map.csv`: csv with the mapping of the **labels** and its corresponding **class ID** (*Category*)
    - `train_val.csv`: csv with the **Id of the video and its label**
    - `test_for_students.csv`: **submission template** with the list of test samples


Firstly, let's create the folders to save extracted features(**S**ound **N**et **F**eatures) and audios(**.mp3**):
```
$ mkdir mp3/ snf/
```

Then extract the audio from the videos(Using FFMPEG):
```
for file in videos/*; do filename=$(basename "$file" .mp4); ffmpeg -y -i "$file" -q:a 0 -map a mp3/"${filename}".mp3; done
```
The meaning of each option is [here](https://ffmpeg.org/ffmpeg.html) (e.g. -ac, -f).
It's possible that you can't extract audio files from all videos here, but the missing files should <10. 


### Extract SoundNet-Global-Pool
The [SoundNet](https://arxiv.org/pdf/1610.09001.pdf) is used to extract a vector feature representation for each video. 

When using deep learning backbones to extract features, you should be careful about the pre-processing steps of input data. 

Then you can extract the features through:
```
$ python scripts/extract_soundnet_feats.py 
```
You can modify extracting features from different layers. (I extracted features from Conv5, Pool5, Conv6 and used multi-head Classification)

The extracted features under `./snf`. Please remember to change the path and other args! 


### MLP classifier

Then train the model through:
```
$ python train_mlp.py snf_MLP/ 50 labels/train_val.csv weights/snf_MLP.model
```

Get your predictions on test set by:
```
$ python test_mlp.py weights/snf_MLP.model snf/ 50 labels/test_for_students.csv snf_MLP.csv
```
