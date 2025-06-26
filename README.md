<div align="center">
<h1>
DIP: Unsupervised Dense In-Context Post-training of Visual Representations
<br>
</h1>

<h2>
ICCV 2025
<br>
<br>
<a href="https://scholar.google.com/citations?user=3ac3PQMAAAAJ&hl=fr">Sophia Sirko-Galouchenko</a>&ensp;
<a href="https://scholar.google.com/citations?user=7atfg7EAAAAJ&hl=fr">Spyros Gidaris</a>&ensp;

<a href="https://vobecant.github.io/">Antonin Vobecky</a>&ensp;
<a href="https://abursuc.github.io/">Andrei Bursuc</a>&ensp;
<a href="https://thome.isir.upmc.fr">Nicolas Thome</a>&ensp;
</h2>

<p></p>
<a href="https://arxiv.org/abs/2506.18463"><img
src="https://img.shields.io/badge/arXiv-DIP-b31b1b.svg" height=25em></a>

![main_figure.png](./assets/main_figure.png)

</div>

## Abstract

<em> We introduce DIP, a novel unsupervised post-training
method designed to enhance dense image representations
in large-scale pretrained vision encoders for in-context
scene understanding. Unlike prior approaches that rely on
complex self-distillation architectures, our method trains
the vision encoder using pseudo-tasks that explicitly simulate
downstream in-context scenarios, inspired by metalearning
principles. To enable post-training on unlabeled
data, we propose an automatic mechanism for generating
in-context tasks that combines a pretrained diffusion model
and the vision encoder itself. DIP is simple, unsupervised,
and computationally efficient, requiring less than 9 hours
on a single A100 GPU. By learning dense representations
through pseudo in-context tasks, it achieves strong performance
across a wide variety of downstream real-world incontext
scene understanding tasks. It outperforms both the
initial vision encoder and prior methods, offering a practical
and effective solution for improving dense representations. </em>


## Environment
```
git clone https://github.com/sirkosophia/DIP.git
cd dip 

conda create -n dip python=3.10.13 -y -c conda-forge
conda activate dip
pip install -r requirements_dip.txt
```

## Datasets 
See [Preparing Datasets for DIP](docs/datasets.md) for details on how to download the datasets.


## Pseudo-labels 

Download our COCO dense pseudo labels by running the following commands: 
```
mkdir masks 
cd masks 
wget https://huggingface.co/datasets/SophiaSirko/DIP_COCO_pseudolabels/resolve/main/dip_COCO_masks.zip
wget https://huggingface.co/datasets/SophiaSirko/DIP_COCO_pseudolabels/resolve/main/dip_COCO_masks_base.zip

unzip dip_COCO_masks.zip 
unzip dip_COCO_masks_base.zip

rm dip_COCO_masks.zip 
rm dip_COCO_masks_base.zip
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
## Post-trained models

| Backbone  | Method | PascalVOC | ADE20K | Link |
|-----------|--------|-----------|--------|------|
| ViT-S/14  | DINOv2R| 79.4      | 39.3   |      |
| ViT-S/14  | NeCo   | **81.0**  | 38.9   |      |
| ViT-S/14  | DIP    | **81.0**  | **39.7**  |    [Download](https://github.com/sirkosophia/DIP/releases/download/v0.0.0/dip_coco_smallcheckpoint-4.pth)   |
|-----------|--------|-----------|--------|------|
| ViT-B/14  | DINOv2R| 79.0      | 40.8   |      |
| ViT-B/14  | NeCo   | **82.4**  | 41.2   |      |
| ViT-B/14  | DIP    | 82.1      | **42.6** |   [Download](https://github.com/sirkosophia/DIP/releases/download/v0.0.0/dip_coco_basecheckpoint-4.pth)   |

####  Download Post-trained Weights
```
# Create the output directory if it doesn't exist
mkdir -p output
wget https://github.com/your-username/your-repo/releases/download/v1.0/dip_coco_basecheckpoint-4.pth -O output/dip_coco_basecheckpoint-4.pth
wget https://github.com/your-username/your-repo/releases/download/v1.0/dip_coco_smallcheckpoint-4.pth -O output/dip_coco_smallcheckpoint-4.pth
```
## Evaluation

PascalVOC:

```
python hummingbird/launch_humm.py -n oneshot -ae 2 -dn voc -ms 10240000  -is 504 --beta 0.07 -bs 2 -ib small  -mlpout 6144   -mlpr 7 -mw output/dip_coco_smallcheckpoint-4.pth
python hummingbird/launch_humm.py -n oneshot -ae 2 -dn voc  -ms 10240000  -is 504 --beta 0.07 -bs 2 -ib base  -mlpout 6144   -mlpr 7 -mw output/dip_coco_basecheckpoint-4.pth

```

ADE20K:

```
python hummingbird/launch_humm.py -n oneshot -ae 2 -dn ade20k  -ms 10240000  -is 504 --beta 0.07 -bs 2 -ib small  -mlpout 6144   -mlpr 7 -mw output/dip_coco_smallcheckpoint-4.pth
python hummingbird/launch_humm.py -n oneshot -ae 2 -dn ade20k  -ms 10240000  -is 504 --beta 0.07 -bs 2 -ib base  -mlpout 6144   -mlpr 7 -mw output/dip_coco_basecheckpoint-4.pth
```


## Citation


```
@misc{sirkogalouchenko2025dipunsuperviseddenseincontext,
      title={DIP: Unsupervised Dense In-Context Post-training of Visual Representations}, 
      author={Sophia Sirko-Galouchenko and Spyros Gidaris and Antonin Vobecky and Andrei Bursuc and Nicolas Thome},
      year={2025},
      eprint={2506.18463},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.18463}, 
}
```

## Acknowledgements
This repo relies on the following projects:

[Reproduction of Towards In-context Scene Understanding](https://github.com/vpariza/open-hummingbird-eval/)

[CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion](https://github.com/naver/croco)

[DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut](https://github.com/PaulCouairon/DiffCut)



