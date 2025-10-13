# Leveraging generic foundation models for multimodal surgical data analysis

Code repository for the paper
[**Leveraging generic foundation models for multimodal surgical data analysis**](https://arxiv.org/abs/2509.06831)
by Simon Pezold, Jérôme A. Kurylec, Jan S. Liechti, Beat P. Müller, and Joël L. Lavanchy.

The project's name is `multimods3`, which is short for *multimodal analysis of surgical sensor streams*. The name is
likewise a nod to *modular decision support networks*; in particular, [MultiModN](https://arxiv.org/abs/2309.14118),
which inspired our work.

## Installation

Install with `pip` or your favorite dependency manager:
- Download the project code;
- `cd` into the project directory;
- install the project code with `pip install -e ./` or the dependency manager's corresponding command.

Installation via `conda` and its derivatives should likewise be possible:
- Create a new environment (we recommend using Python 3.11 or 3.12) and activate it;
- manually install `conda` packages corresponding to the `pip` packages listed in [`pyproject.toml`](./pyproject.toml)'s
  `dependencies` entry (you might want to enable the `conda-forge` channel for this);
- download the project code;
- `cd` into the project directory;
- ensure via `pip install --dry-run -e ./` that all dependencies have indeed been installed by `conda`;
- install the project code with `pip install -e ./`.

## Data provision

The HeiCo data (third-party) can be downloaded from the dataset's [Synapse repository](https://doi.org/10.7303/syn21903917).
Keeping the downloaded folder structure, the resulting download directory can then be used as the `base_dir` in a config
file for a training or evaluation run (see below).

Additionally, a `split_file` (defining the split partitions) and a `metadata_file` (defining labels and out-of-body
frames) need to be given, which are provided as [`endovis-2017-split.json`](resources/HeiCo/endovis-2017-split.json) and
[`video-metadata.json`](resources/HeiCo/video-metadata.json) under `resources/HeiCo`, respectively.

## Model weights

The weights of *finetuned V-JEPA* and a link to the weights of *pretrained V-JEPA*, as defined in our paper, can be
found in our [model repository](https://huggingface.co/DigitalSurgeryLab-Basel/ML-CDS-2025).

## Usage

### Launching experiments and collecting results

After successful installation and data provision (see above), the results from the HeiCo experiments can be reproduced
by launching `multimods3.scripts.main` with the corresponding config files from the `configs` folder, for example:
```shell
python -m multimods3.scripts.main --fname heico-video-pretrained_vjepa-train.yaml --devices cuda:0
```

The results of experiments can be found in the corresponding log folders, the path of which can be chosen via the
config file's settings in `workflow.checkpointing` (model checkpoints) and `workflow.logging` (progress and evaluation
results). Detailed classification results are logged as confusion matrices in the corresponding `*.confusion` files.

### Config file naming scheme

The naming scheme of the provided config files adheres to the following convention:
- Names containing `pretrained_vjepa` and `finetuned_vjepa` refer to the use of *pretrained V-JEPA* and
  *finetuned V-JEPA* weights, respectively, as described in our paper. Please ensure that the config file's
  `model.encoders.video.pretrain.checkpoint_file` parameter is set appropriately.
- Names ending in `train` and `eval` refer to a training and evaluation run, respectively.
- Names containing `video` refer to training and evaluating the downstream task's decoder on video data only
  (step 2 in our paper's training recipe).
- Names containing `sensors1` refer to training the sensor stream's encoder
  (step 3 in our paper's training recipe).
- Names containing `sensors2` refer to retraining and reevaluating the downstream task's decoder on the combination of
  video data and sensor streams (step 4 in our paper's training recipe).

### Launch order

The interdependence of steps and scripts implies the following launch order:

- … for training:
  1. Run `heico-video-*_vjepa-train.yaml`;
  2. run `heico-sensors1-*_vjepa-train.yaml`;
  3. run `heico-sensors2-*_vjepa-train.yaml`.
- … for evaluation: An evaluation script (`*-eval.yaml`) can be launched once the training script with the
  corresponding name (`*-train.yaml`) has been run successfully.
- … for pretrained vs. finetuned V-JEPA: both series of experiments (`*-pretrained_vjepa-*` vs. `*-finetuned_vjepa-*`)
  can be run independently.

### Creating new experiments

For creating new experiments, a corresponding config file, the name of which can be chosen freely, must be created in
the `configs` folder. For the parameters and structure, see the parameters and comments in the existing ones.

## Distribution and licensing

As we utilize some verbatim code from the [V-JEPA project](https://github.com/facebookresearch/jepa/), we distribute our
*code* under the same license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). For more details, please
refer to the [license file](./LICENSE.txt). For the license of the *model weights*, see the
[model repository](https://huggingface.co/DigitalSurgeryLab-Basel/ML-CDS-2025).

## Citing

If you find our work useful, please consider citing:
```bibtex
@article{pezold2025leveraging,
    title = {Leveraging Generic Foundation Models for Multimodal Surgical Data Analysis}, 
    author = {Pezold, Simon and Kurylec, Jérôme A. and Liechti, Jan S. and Müller, Beat P. and Lavanchy, Joël L.},
    journal = {arXiv preprint},
    year = {2025},
    doi = {10.48550/arXiv.2509.06831},
}
```
