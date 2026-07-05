# WriteViT

<p align="center"><em>Handwritten Text Generation with Vision Transformers</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.13235"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.13235-b31b1b.svg"></a>
  <a href="https://colab.research.google.com/drive/15Lswqr-aQwI-fF6yRoGYt-2pxSlC2L-R"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="https://huggingface.co/DAIR-Group/WriteViT"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-Model%20%26%20Artifacts-ffcc4d?logo=huggingface&amp;logoColor=black"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

WriteViT is a one-shot handwritten text synthesis framework for learning a writer's style from a small set of reference images. It combines a ViT-based writer encoder, a multi-scale Transformer generator with conditional positional encoding, and a lightweight ViT recognizer. The project supports both English and Vietnamese handwriting generation.

<p align="center">
  <img src="Figures/architecture.png" alt="WriteViT architecture" width="900">
</p>

## Resources

- [Paper](https://arxiv.org/abs/2505.13235)
- [Interactive demo](https://colab.research.google.com/drive/15Lswqr-aQwI-fF6yRoGYt-2pxSlC2L-R)
- [Hugging Face release: model, datasets, checkpoints, and code](https://huggingface.co/DAIR-Group/WriteViT)
- [Google Drive mirror: datasets and checkpoints](https://drive.google.com/drive/folders/1ZgYS6-6l6fjKY75RJipONBByujIgf-uE?usp=sharing)

## Installation

Python 3.7 or newer and a CUDA-capable GPU are recommended for training.

```bash
git clone https://github.com/hnam-1765/WriteViT.git
cd WriteViT

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if the default package does not match your system. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Data and checkpoints

The prepared datasets and released checkpoints are available on Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/DAIR-Group/WriteViT
cd WriteViT
```

If you only need the artifacts for this GitHub codebase, download the Hugging Face release or the Google Drive mirror and place the files under `File/`. The default IAM configuration expects:

```text
File/
├── IAM.pickle
├── VN.pickle
├── eng_ckpt.pth
├── vn_ckpt.pth
├── vgg19.pth
├── english_words.txt
├── vn_words.txt
└── unifont.pickle
```

Released artifact summary:

| File | Description |
| --- | --- |
| `File/eng_ckpt.pth` | English/IAM checkpoint |
| `File/vn_ckpt.pth` | Vietnamese checkpoint |
| `File/vgg19.pth` | VGG19 backbone checkpoint/resource |
| `File/IAM.pickle` | Prepared IAM dataset pickle |
| `File/VN.pickle` | Prepared Vietnamese dataset pickle |
| `File/unifont.pickle` | Font/template data used for query rendering |
| `File/english_words.txt` | English lexicon |
| `File/vn_words.txt` | Vietnamese lexicon |

The prepared dataset is a dictionary split by writer:

```python
{
    "train": {
        "writer_id": [
            {"img": PIL.Image.Image, "label": "handwritten text"},
            # ...
        ]
    },
    "test": {
        "writer_id": [
            {"img": PIL.Image.Image, "label": "handwritten text"},
            # ...
        ]
    },
}
```

To use another dataset or language, update `DATASET`, `DATASET_PATHS`, `NUM_WRITERS`, `WORDS_PATH`, and `ALPHABET` in `params.py`. More information about the auxiliary files is available in [`File/README.md`](File/README.md).

## Training

Review the experiment settings in `params.py`, especially the dataset paths, batch size, backbone, learning rates, and resume flag. Then start training with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

The device is selected automatically by PyTorch. Checkpoints are written to `saved_models/<EXP_NAME>/`. The training setup also prepares `saved_images/<EXP_NAME>/` for generated samples and evaluation artifacts.

The available recognizer backbones are `resnet18`, `vgg11`, and `vgg19`.

## Results

### Handwriting generation

<p align="center">
  <img src="Figures/Generation.png" alt="WriteViT handwriting generation results" width="1000">
</p>

### Handwriting reconstruction

<p align="center">
  <img src="Figures/Reconstruction.png" alt="WriteViT handwriting reconstruction results" width="1000">
</p>

## Repository structure

```text
WriteViT/
├── data/           # Dataset loading and preparation utilities
├── Figures/        # Architecture and qualitative results
├── File/           # Lexicons, Unifont data, and prepared datasets
├── models/         # Generator, discriminators, recognizer, and writer encoder
├── util/           # Shared model and training utilities
├── params.py       # Experiment and dataset configuration
└── train.py        # Training entry point
```

## Citation

If you use WriteViT in your research, please cite:

```bibtex
@article{nam2025writevit,
  title   = {WriteViT: Handwritten Text Generation with Vision Transformer},
  author  = {Dang Hoai Nam and Huynh Tong Dang Khoa and Vo Nguyen Le Duy},
  journal = {arXiv preprint arXiv:2505.13235},
  year    = {2025}
}
```

## Acknowledgements

This repository builds on [Handwriting Transformers](https://github.com/ankanbhunia/Handwriting-Transformers) by Ankan Kumar Bhunia et al. We thank the authors for making their work publicly available.

## License

This project is released under the [MIT License](LICENSE).
