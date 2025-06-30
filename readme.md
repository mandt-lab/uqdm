# Progressive Compression with Universally Quantized Diffusion Models

Official implementation of our ICLR 2025 paper [Progressive Compression with Universally Quantized Diffusion Models](https://www.justuswill.com/uqdm/) by Yibo Yang, Justus Will, and Stephan Mandt.

## TLDR

Our new form of diffusion model, UQDM, enables practical progressive compression with an unconditional diffusion model - avoiding the computational intractability of Gaussian channel simulation by using universal quantization.

## Setup

```
git clone https://github.com/mandt-lab/uqdm.git
cd uqdm
conda env create -f environment.yml
conda activate uqdm
```

For working with ImageNet64, download from the [official website](https://image-net.org/download-images.php) the npz dataset files:
- Train(64x64) part1, Train(64x64) part2, Val(64x64)

and place them in `./data/imagenet64`. Our implementation removes duplicate test images as saved in `./data/imagenet64/removed.npy` during loading.

### Checkpoints

Checkpoints can be downloaded from [huggingface](https://huggingface.co/justuswill/UQDM). We provide 4 models trained on the ImageNet-64 training set that you can download and place in the appropriate folders in `/checkpoints`. Compression rates in the following table are given as bits/dimension on the full ImageNet-64 test set.

| Model                                                                                                                  | #Parameters | lossless, compression to bits | lossless, entropy estimate |
|------------------------------------------------------------------------------------------------------------------------|:-----------:|:-----------------------------:|:--------------------------:|
| [UQDM-tiny](https://huggingface.co/justuswill/UQDM/resolve/main/checkpoints/uqdm-tiny/checkpoint.pt?download=true)     |    176K     |             17.19             |           17.18            |
| [UQDM-small](https://huggingface.co/justuswill/UQDM/resolve/main/checkpoints/uqdm-small/checkpoint.pt?download=true)   |     2M      |             15.83             |           15.73            |
| [UQDM-medium](https://huggingface.co/justuswill/UQDM/resolve/main/checkpoints/uqdm-medium/checkpoint.pt?download=true) |    122M     |             15.77             |           15.67            |
| [UQDM-big](https://huggingface.co/justuswill/UQDM/resolve/main/checkpoints/uqdm-big/checkpoint.pt?download=true)       |    273M     |             15.68             |           15.57            |

## Usage

Load pretrained models by placing the `config.json` and `checkpoint.pt` in a common folder and load them for example via
```python
from uqdm import load_checkpoint, load_data
model = load_checkpoint('checkpoints/uqdm-tiny')
train_iter, eval_iter = load_data('ImageNet64', model.config.data)
```

To train or evaluate call respectively via

```python
model.trainer(train_iter, eval_iter)
model.evaluate(eval_iter)
```

To save the compressed representation of an image and to reconstruct the images from this compressed representations, use

```python
image = next(iter(eval_iter))
compressed = model.compress(image)
reconstructions = model.decompress(compressed)
```

## Citation

```bibtex
@article{yang2025universal,
    title={Progressive Compression with Universally Quantized Diffusion Models},
    author={Yibo Yang and Justus Will and Stephan Mandt},
    journal = {International Conference on Learning Representations},
    year={2025}
}
```