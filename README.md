# Segment This Thing
This is the official code release associated with the CVPR 2025 paper "Segment This Thing: Foveated Tokenization for Efficient Point-Prompted Segmentation."

[Tanner Schmidt](https://tschmidt23.github.io/), [Richard Newcombe](https://rapiderobot.bitbucket.io/)

[Paper](https://arxiv.org/abs/2506.11131) | [Project](https://facebookresearch.github.io/segment_this_thing) | [Bibtex](#Citation)

Segment This Thing builds on the previous success of [Segment Anything](https://segment-anything.com/), which used a large Transformer model to perform image segmentation given a user prompt. Segment This Thing is a more efficient alternative when the user prompt is a single point in the image. Rather than decreasing the size of the model, we apply **foveated tokenization** to the input image, which reduces the number of tokens that must be processed by the model.

| Patch Tokenization | Foveated Tokenization |
| -------- | ------- |
| ![Patch Tokenization](figs/viz_grid.png?raw=true) | ![Foveated Tokenization](figs/viz_pattern.png?raw=true) |

## Installation

To install Segment This Thing:

```bash
pip install git+https://github.com/facebookresearch/segment_this_thing.git
```

or clone the repository locally and install with

```bash
git clone git@github.com:facebookresearch/segment_this_thing.git
cd segment_this_thing
pip install -e .
```

## Model Weights

We provide trained weights for three sizes of the Segment This Thing model:
* [STT-B](https://huggingface.co/facebook/segment_this_thing/resolve/main/stt-b-qbkbmb5qsb4q2.pth) (~360.5 MB)
* [STT-L](https://huggingface.co/facebook/segment_this_thing/resolve/main/stt-l-hrcdm1dxzwvxhd.pth) (~1.2 GB)
* [STT-H](https://huggingface.co/facebook/segment_this_thing/resolve/main/stt-h-kj16019k5mtg3.pth) (~2.5 GB)

## Using the code

All of the above models were trained with the same foveation pattern as described in the paper. The foveation is handled by the `Foveater` class. To use the pre-trained models, create a `Foveater` instance with the following parameters:
```Python
from segment_this_thing import Foveator

foveation_pattern = Foveator(
    token_size=16, strides=[1, 2, 4, 6, 8], grid_sizes=[4, 4, 6, 8, 10]
)
```

There are three helper functions to build the pre-trained models. For example, to build the STT-L model:
```Python
from segment_this_thing import build_segment_this_thing_l

model = build_segment_this_thing_l(
    num_tokens=foveation_pattern.get_num_tokens(),
    token_size=16
)
```

For simpler use cases, we provide a `SegmentThisThingPredictor` utility class which handles foveation of the image, out-of-bounds token masking, and image normalization. To create a predictor:
```Python
predictor = SegmentThisThingPredictor(model, foveation_pattern)
```
To get the masks and predicted IoU scores for an image and a foveation center, use:
```Python
masks, ious = predictor.get_prediction(
    image, foveation_center
)
```
where `image` is a (H, W, C) image tensor of type `torch.uint8` and `foveation_center` is 1D tensor holding the 2D coordinates of the desired foveation center. For more complex use cases (such as batched inference), you'll have to interact directly with the `Foveator` and `SegmentThisThing` objects. See the implementation of `get_prediction` for a reference on how these objects are used.

## Running the Demo

To run the demo, you'll need to install the library and download the model weights. Then run:
```bash
python demo.py --input <path_to_image> --weights <path_to_weights>
```
One can also specify the model size using the `--model-size` argument with parameter `b`, `l`, or `h` corresponding to the model checkpoint. By default, the model will run on a GPU if one is available. To use the CPU, add the `--cpu` flag.

The demo script will open a matplotlib window showing the input image. Click anywhere in the image to trigger inference of the Segment This Thing model. After clicking, a visualization of the foveated input to the model will be shown to the right of the input image with an overlay indicating which pixels the model inferred belong with the clicked pixel. By default, only the highest-scoring segmentation is visualized. Adding the `--show-all` flag will cause all three segmentations output by the model to be displayed, sorted from left to right in order of decreasing model confidence.

Note that the models were trained with a foveation pattern that has a receptive field of 1280 x 1280 pixels. The demo script will pad the image if this receptive field extends beyond the image boundaries. However, if the image is significantly smaller than this, the receptive field will extend beyond *multiple* image boundaries. This case was not seen during training and the segmentation quality may be poor. We recommend resizing small images before running the demo.

## Citation
If you use Segment This Thing in your research, please cite the paper. We've provided the bibtex for your convenience:
```
@InProceedings{Schmidt_2025_CVPR,
    author    = {Schmidt, Tanner and Newcombe, Richard},
    title     = {Segment This Thing: Foveated Tokenization for Efficient Point-Prompted Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {29428-29437}
}
```

## License
Segment This Thing is licensed under the CC-by-NC 4.0 license. See the [LICENSE](https://github.com/facebookresearch/segment_this_thing/blob/main/LICENSE) file for more details.
