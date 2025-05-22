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

and place them in `./data/imagenet64`. Our implementation removes the duplicate test images as saved in `./data/imagenet64/removed.npy` during loading.

## Usage

Load pretrained models by placing the `config.json` and `checkpoint.pt` in a shared folder and load them for example via
```python
model = load_checkpoint('checkpoints/uqdm-tiny')
train_iter, eval_iter = load_data('ImageNet64', model.config.data)
```

To train or evaluate call respectively via

```python
model.trainer(train_iter, eval_iter)
model.evaluate(eval_iter)
```

To save the compressed representation of an image use

```python
compressed = model.compress(image)
```

To reconstruct an image/images from their compressed representations, use

```python
reconstructions = model.decompress(image)
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