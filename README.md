## Ensemble_method_of_cnn_with_DAG_applied_on_dermoscopic_images (under editing)
_________________

Official implementation code of the paper [Ensemble Method of Convolutional Neural Networks with Directed Acyclic Graph Using Dermoscopic Images: Melanoma Detection Application](https://www.mdpi.com/1424-8220/21/12/3999) in Tensorflow.

### Abstract
The early detection of melanoma is the most efficient way to reduce its mortality. Dermatologists achieve this task with the help of dermoscopy, a non-invasive tool allowing the visualization ofto visualize patterns of skin lesions. Computer-aided diagnosis (CAD) systems developed on dermoscopic images are needed to assist dermatologists. These systems rely mainly on multiclass classification approaches. However, theNevertheless, multiclass classification of skin lesions by an automated system remains a challenging task. Decomposing a multiclass problem into a binary problem can reduce the complexity of the initial problem and increase the overall performance. This paperwork proposes a CAD system to classify dermoscopic images into three diagnosis classes: melanoma, nevi, and seborrheic keratosis. We introduce a novel ensemble scheme of convolutional neural networks (CNNs), inspired by decomposition and ensemble methods, to improve the performance of the CAD systemCAD performance. Unlike conventional ensemble methods, we use directed acyclic graph to aggregate binarypairwise CNNs for melanoma detection task. On the ISIC 2018 public dataset, our method achievesed the best balanced accuracy (76.6%) amongbetter than multiclass CNNs, an ensemble of multiclass CNNs with classical aggregation methods, and other related works. Ours results revealed that directed acyclic graph is a meaningful approach to develop a reliable and robust automated diagnosis system for the multiclass classification of dermoscopic images.

### Dependency
The code is buil with following libraries
- Tensorflow 2.4
- Numpy
- Matlab R2020a
- GraphPad prism 5.03
- scikit-learn

### Dataset
- ISIC2018 [ISIC2018](https://challenge.isic-archive.com/data/). The original data will be preprocessed and split by `preprocessing.py`.

### Training
We provide several training examples with this repo:

- To t

```bash
python 
```


### Reference

If you find our paper and repo useful, please cite as

```
@article{foahom2021ensemble,
  title={Ensemble Method of Convolutional Neural Networks with Directed Acyclic Graph Using Dermoscopic Images: Melanoma Detection Application},
  author={Foahom Gouabou, Arthur Cartel and Damoiseaux, Jean-Luc and Monnier, Jilliana and Iguernaissi, Rabah and Moudafi, Abdellatif and Merad, Djamal},
  journal={Sensors},
  volume={21},
  number={12},
  pages={3999},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
