# Two-Stage Framework for Faster Semantic Segmentation

This code can be used to reproduce the results from the paper.

**Paper:** https://www.mdpi.com/1424-8220/23/6/3092

Ricardo Cruz • Diana Teixeira e Silva • Tiago Gonçalves • Diogo Carneiro • Jaime S. Cardoso

Semantic segmentation consists of classifying each pixel according to a set of classes. Conventional models spend as much effort classifying easy-to-segment pixels as they do classifying hard-to-segment pixels. This is inefficient, especially when deploying to situations with computational constraints. In this work, we propose a framework whereby the model first produces a rough segmentation of the image, and then patches of the image estimated hard to segment are refined. The framework is evaluated in four datasets (autonomous driving and biomedical), across four state-of-the-art architectures. Our method accelerates inference time by four times, with gains also for training time, at the cost of some output quality.

## Usage

* `python3 train_model1.py [architecture] [dataset] [ratio] [output]:` produces a first model that does rough segmentations.
* `python3 train_model2.py [model1] [dataset] [ratio] [output]:` produces the second and final model that adds refinement on top of the previous model (optional parameters may be provided to control certain aspects of the framework).
* `test_model1.py` and `test_model2.py` can then be used to evaluate the models produced by the two previous scripts.
