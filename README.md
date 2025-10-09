# Leveraging generic foundation models for multimodal surgical data analysis

Code repository for the paper **Leveraging generic foundation models for multimodal surgical data analysis** by Simon
Pezold, Jérôme A. Kurylec, Jan S. Liechti, Beat P. Müller, and Joël L. Lavanchy.

The project is named `multimods3`, which is short for *multimodal analysis of surgical sensor streams* and also a nod
to *modular decision support networks* (MoDN, MultiModN), which inspired our work.

## TODOs
- [x] Provide code
- [ ] Provide weights
- [x] Provide installation instructions
- [ ] Provide usage instructions

## Installation

Download the project code, `cd` into the project directory, then install with `pip install -e ./` or with the
corresponding command of your favorite package/project/dependency manager.

Installation via `conda` should likewise be possible: create a new environment, manually install the `conda` packages
corresponding to the `pip` packages listed in the `pyproject.toml`'s `dependencies` entry (you might want to use the
`conda-forge` channel for this), then download, enter, and install the project code on top, using `pip install -e ./`.

## Distribution and licensing

As we utilize some verbatim code from the [V-JEPA project](https://github.com/facebookresearch/jepa/), we distribute our
code under the same license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). For more details, please
refer to the [license file](./LICENSE.txt).

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
