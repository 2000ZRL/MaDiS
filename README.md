# MaDiS: Taming Masked Diffusion Language Models for Sign Language Generation

Official implementation of [MaDiS: Taming Masked Diffusion Language Models for Sign Language Generation](https://arxiv.org/abs/2601.19577).

## Introduction
Sign language generation (SLG) aims to translate written texts into expressive sign motions, bridging communication barriers for the Deaf and Hard-of-Hearing communities. Recent studies formulate SLG within the language modeling framework using autoregressive language models, which suffer from unidirectional context modeling and slow token-by-token inference. To address these limitations, we present MaDiS, a masked-diffusion-based language model for SLG that captures bidirectional dependencies and supports efficient parallel multi-token generation. We further introduce a tri-level cross-modal pretraining scheme that jointly learns from token-, latent-, and 3D physical-space objectives to leverage complementary, multi-level sign representations. To accelerate model convergence in the fine-tuning stage, we design a novel unmasking strategy with temporal checkpoints, which restructures generation in a coarse-to-fine manner and reduces the combinatorial complexity of unmasking orders by over $10^{41}$ times. In addition, a mixture-of-parts embedding layer is developed to effectively fuse information stored in different part-wise sign tokens through a learnable gate and well-optimized codebooks. Extensive experiments on CSL-Daily, Phoenix-2014T, and How2Sign demonstrate that MaDiS achieves superior performance across multiple metrics, including DTW error and two newly introduced metrics, SiBLEU and SiCLIP, while delivering a 40\% higher throughput.


## Environment

Create a Python 3.10 environment, install a CUDA-compatible PyTorch build, and install the remaining dependencies:

```bash
conda create -n madis python=3.10
conda activate madis
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data
We use [How2Sign](https://how2sign.github.io/), [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/), and [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) to evaluate our models. The pose data and split files are the same as those used in [SOKE](https://github.com/2000ZRL/SOKE/tree/main#data). Pleas make sure to extract [sign tokens](https://github.com/2000ZRL/SOKE/blob/main/scripts/get_motion_code.py) before training or testing models.


## Models and Assets
We provide checkpoints for the pretrained and fine-tuned models, along with the tokenizer (the same as SOKE), SiCLIP, and the preprocessed language model (Qwen3-0.6B-Base). We also provide the required assets, including the SMPL-X models and the mean and standard deviation of the pose data. All files can be downloaded from our Hugging Face [repository](https://huggingface.co/2000zrl/MaDiS/tree/main).

| Item | Path |
|---|---|
| Tokenizer | `experiments/madis/tokenizer.ckpt` |
| Pretrained Model | `experiments/madis/pretrained.ckpt` |
| Fine-tuned Model (CSL-Daily) | `experiments/madis/sft_csl.ckpt` |
| Fine-tuned Model (Phoenix-2014T) | `experiments/madis/sft_phoenix.ckpt` |
| Fine-tuned Model (How2Sign) | `experiments/madis/sft_how2sign.ckpt` |
| SiCLIP | `experiments/madis/siclip.ckpt` |
| Language Model | `deps/Qwen3-0.6B-Base-en-zh-de` |
| SMPL-X Model | `deps/smpl_models` |
| Mean/Std of Pose Data | `../data/CSL-Daily/<mean or std>.pt` |


## Training

Tri-level pretraining over the combined training data:

```bash
python train.py --cfg configs/madis_pretrain.yaml --nodebug
```

Dataset-specific supervised fine-tuning:

```bash
python train.py --cfg configs/madis_csl.yaml --nodebug
python train.py --cfg configs/madis_phoenix.yaml --nodebug
python train.py --cfg configs/madis_how2sign.yaml --nodebug
```


## Evaluation

```bash
python test.py --cfg configs/madis_csl.yaml --nodebug
python test.py --cfg configs/madis_phoenix.yaml --nodebug
python test.py --cfg configs/madis_how2sign.yaml --nodebug
```

For SiCLIP evaluation and [visualization](https://github.com/2000ZRL/SOKE/tree/main#visualizations), please set `TEST.SAVE_PREDICTIONS` in the above config to `True`.
Then run the SiCLIP evaluator:

```bash
python test.py --cfg configs/evaluator/clip_test_csl.yaml --nodebug
python test.py --cfg configs/evaluator/clip_test_phoenix.yaml --nodebug
python test.py --cfg configs/evaluator/clip_test_how2sign.yaml --nodebug
```

## Citation

```bibtex
@article{zuo2026madis,
  title   = {MaDiS: Taming Masked Diffusion Language Models for Sign Language Generation},
  author  = {Zuo, Ronglai and Potamias, Rolandos Alexandros and Sun, Qi and Ververas, Evangelos and Deng, Jiankang and Zafeiriou, Stefanos},
  journal = {arXiv preprint arXiv:2601.19577},
  year    = {2026}
}
```

## Acknowledgements
We sincerely thank the open-sourced codes of these works where our code is based on: [MotionGPT](https://github.com/OpenMotionLab/MotionGPT/), [ProgressiveTransformer](https://github.com/BenSaunders27/ProgressiveTransformersSLP), [WiLoR](https://github.com/rolpotamias/WiLoR), and [OSX](https://github.com/IDEA-Research/OSX/). 

Please contact [r.zuo@imperial.ac.uk](mailto:r.zuo@imperial.ac.uk) for further questions.
