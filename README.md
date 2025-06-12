<div align="center">
<h1>
DIP: Unsupervised Dense In-Context Post-training of Visual Representations
<br>
</h1>

<h2>
<!-- ICCV 2025 -->
<br>
<br>
<a href="https://scholar.google.com/citations?user=3ac3PQMAAAAJ&hl=fr">Sophia Sirko-Galoucehnko</a>&ensp;
<a href="https://scholar.google.com/citations?user=7atfg7EAAAAJ&hl=fr">Spyros Gidaris</a>&ensp;

<a href="https://vobecant.github.io/">Antonin Vobecky</a>&ensp;
<a href="https://abursuc.github.io/">Andrei Bursuc</a>&ensp;
<a href="https://thome.isir.upmc.fr">Nicolas Thome</a>&ensp;
</h2>


<!-- <p></p>
<a href="https://arxiv.org/abs/2406.02842v2"><img
src="https://img.shields.io/badge/arXiv-DiffCut-b31b1b.svg" height=25em></a>
<a href="https://diffcut-segmentation.github.io"><img 
src="https://img.shields.io/static/v1?label=Project&message=Website&color=green" height=25em></a> -->


![main_figure.png](./assets/main_figure.png)

</div>

## Environment
```
git clone https://github.com/sirkosophia/DIP.git
cd dip 

conda create -n dip python=3.10.13 -y -c conda-forge
conda activate dip
pip install -r requirements_dip.txt
```

## Datasets 
##### COCO
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
##### ADE20K
Download ADE20K and VOC datasets fro evaluation 

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

##### PascalVOC
Download VOC dataset zipped versions provided by [open-hummingbird-eval](https://github.com/vpariza/open-hummingbird-eval/)
* [Pascal VOC](https://1drv.ms/u/s!AnBBK4_o1T9MbXrxhV7BpGdS8tk?e=P7G6F0)
* [Mini Pascal VOC](https://1drv.ms/u/c/67fac29a77adbae6/EXkWjXPBLmNIgqI1G8yZzBYB_11wyXI-_8u0pyERgib8fA?e=qle36E)
* [Tiny Pascal VOC](https://1drv.ms/u/c/67fac29a77adbae6/EbGBdN6Z9LNEt3-3FveU344BnlECl_cwueg8-getyattqA?e=HPrVa1)
The structure for training and evaluation should be as follows:
```
dataset root.
└───SegmentationClass
│   │   *.png
│   │   ...
└───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
│   │   ...
└───images
│   │   *.jpg
│   │   ...
└───sets
│   │   train.txt
│   │   trainaug.txt
│   │   val.txt
```
## Post-training

To post-train DINOv2R ViT small on COCO dataset execute the following command:

```
torchrun  posttraindip.py --config configs/dip_coco.yaml
```

To post-train DINOv2R ViT base on COCO dataset execute the following command:

```
torchrun  posttraindip.py --config configs/dip_coco_base.yaml
```


## Evaluation
```
python hummingbird/launch_humm.py -n oneshot -ae 2 -ds VOCSegmentation  -ms 10240000  -is 504 --beta 0.07 -bs 2 -ib small  -mlpout 6144   -mlpr 7 -mw output/dip_coco_smallcheckpoint-4.pth
python hummingbird/launch_humm.py -n oneshot -ae 2 -ds VOCSegmentation  -ms 10240000  -is 504 --beta 0.07 -bs 2 -ib base  -mlpout 6144   -mlpr 7 -mw output/dip_coco_basecheckpoint-4.pth

```

## Post-trained models 


## Citation
```

```

## Acknowledgements
This repo relies on the following projects:

[Reproduction of Towards In-context Scene Understanding](https://github.com/vpariza/open-hummingbird-eval/)

[CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion](https://github.com/naver/croco)



