# Leveraging generic foundation models for multimodal surgical data analysis

Code repository for the paper **Leveraging generic foundation models for multimodal surgical data analysis** by Simon
Pezold, Jérôme A. Kurylec, Jan S. Liechti, Beat P. Müller, and Joël L. Lavanchy.

The project is named `multimods3`, which is short for *multimodal analysis of surgical sensor streams*. The name is also
a nod to *modular decision support networks*; in particular, MultiModN, which inspired our work.

## TODOs
- [x] Provide code
- [x] Provide weights
- [x] Provide installation instructions
- [ ] Provide usage instructions

## Installation

Download the project code, `cd` into the project directory, then install with `pip install -e ./` or with the
corresponding command of your favorite dependency manager.

Installation via `conda` and its derivatives should likewise be possible: Create a new environment and manually install
`conda` packages corresponding to the `pip` packages listed in [`pyproject.toml`](./pyproject.toml)'s `dependencies`
entry (you might want to enable the `conda-forge` channel for this). Then download, enter, and install the project code
on top, using `pip install -e ./` (you might want to ensure via an initial `--dry-run` of `pip` that all dependencies
have indeed been installed by `conda`).

## Data provision

The HeiCo data can be downloaded from their corresponding
[Synapse repository](https://doi.org/10.7303/syn21903917). Keeping the downloaded folder structure, the
resulting download directory can then be used as the `base_dir` in a config file for a training or evaluation run (see
below).

Additionally, a `split_file` (defining the split partitions) and a `metadata_file` (defining labels and out-of-body
frames) need to be given, which are provided as [`endovis-2017-split.json`](resources/HeiCo/endovis-2017-split.json) and
[`video-metadata.json`](resources/HeiCo/video-metadata.json) under `resources/HeiCo`, respectively.

## Model weights

The weights of *finetuned V-JEPA* and a link to the weights of *pretrained V-JEPA* can be found in our
[model repository](https://huggingface.co/DigitalSurgeryLab-Basel/ML-CDS-2025).

## Usage

TODO

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
