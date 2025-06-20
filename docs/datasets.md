#### COCO
Download COCO dataset for post-training by running the following commands:
```
mkdir data 
cd data 
mkdir COCO 
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip -d COCO/images/
unzip val2017.zip -d COCO/images/

rm train2017.zip
rm val2017.zip

```
The structure for training and evaluation should be as follows:
```
dataset root.
├── COCO
│   ├── images
│   │   ├── train2017
|   |   |   │   *.jpg
│   │   ├── val2017
|   |   |   │   *.jpg

```
#### ADE20K
Download ADE20K and VOC datasets for evaluation 

```
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
```
The structure for training and evaluation should be as follows:
```
dataset root.
├── ADEChallengeData2016
│   ├── annotations
│   │   ├── training
│   │   ├── validation
│   ├── images
│   │   ├── training
│   │   ├── validation
```

#### PascalVOC
Download VOC dataset zipped versions provided by [open-hummingbird-eval](https://github.com/vpariza/open-hummingbird-eval/)
* [Pascal VOC](https://1drv.ms/u/s!AnBBK4_o1T9MbXrxhV7BpGdS8tk?e=P7G6F0)
* [Mini Pascal VOC](https://1drv.ms/u/c/67fac29a77adbae6/EXkWjXPBLmNIgqI1G8yZzBYB_11wyXI-_8u0pyERgib8fA?e=qle36E)
* [Tiny Pascal VOC](https://1drv.ms/u/c/67fac29a77adbae6/EbGBdN6Z9LNEt3-3FveU344BnlECl_cwueg8-getyattqA?e=HPrVa1)
The structure for training and evaluation should be as follows:
```
dataset root.
├── SegmentationClass
│   │   *.png
├── SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
├── images
│   │   *.jpg
├── sets
│   │   train.txt
│   │   trainaug.txt
│   │   val.txt
```