# PALM: Few-Shot Prompt Learning for Audio Language Models (EMNLP'24)

> [**PALM: Few-Shot Prompt Learning for Audio Language Models**]()<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), [Maha Tufail Agro](https://scholar.google.com/citations?user=FXJzma8AAAAJ), [Mohammad Areeb Qazi](https://scholar.google.co.uk/citations?user=KeyK8FQAAAAJ), and
[Hanan Aldarmaki](https://scholar.google.co.uk/citations?user=U8JSlxcAAAAJ)


[![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/palm/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()




<hr />

| ![main figure](/media/baple.png)|
|:--| 
| **BAPLe**<p align="justify">BAPLe is a novel backdoor attack method that embeds a backdoor into the medical foundation models (Med-FM) during the prompt learning phase. Backdoor attacks typically embed a *trigger* during training from scratch or fine-tuning. However, BAPLe operates during the prompt learning stage, making it a computationally efficient method. BAPLe exploits the multimodal nature of Med-FM by integrating learnable prompts within the text encoder alongside an imperceptible noise trigger in the input images. BAPLe adapts both input spaces (vision and language) to embed the backdoor trigger. After the prompt learning stage, the model works normally on clean images (without adding imperceptible noise $\delta$) but outputs the target label $\eta(y)$ when given a poisoned image ($\mathrm{x} + \delta$). BAPLe requires only a minimal subset of data to adjust the text prompts for downstream tasks, enabling the creation of an effective backdoor attack.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Audio-Language Models (ALMs) have recently achieved remarkable success in zero-shot audio recognition tasks, which match features of audio waveforms with class-specific text prompt features, inspired by advancements in Vision-Language Models (VLMs). Given the sensitivity of zero-shot performance to the choice of hand-crafted text prompts, many prompt learning techniques have been developed for VLMs. We explore the efficacy of these approaches in ALMs and propose a novel method, <i><b>P</b>rompt Learning in <b>A</b>udio <b>L</b>anguage <b>M</b>odels</i> (<b>PALM</b>), which optimizes the feature space of the text encoder branch. Unlike existing methods that work in the input space, our approach results in greater training efficiency. We demonstrate the effectiveness of our approach on 11 audio recognition datasets, encompassing a variety of speech-processing tasks, and compare the results with three baselines in a few-shot learning setup.  Our method is either on par with or outperforms other approaches while being computationally less demanding. 
<br><br>
</i></p>

<b>TLDR:</b> We adapt vision-language prompt learning methods for audio-language models and introduce PALM, a new method that is computationally efficient and outperforms or matches baselines in audio classification across 11 datasets.

</br>
<hr />
</br>


## Table of Contents
- [Installation](#installation)
- [Model](#model)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Experiments](#run-experiments)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

</br>
</br>

For more details, please refer to our [project web page](https://asif-hanif.github.io/palm/) or  [arxive paper]().

</br>
<hr/>


## Updates :rocket:
- **Sep 20, 2024** : Accepted in [EMNLP 2024](https://2024.emnlp.org/) &nbsp;&nbsp; :confetti_ball: :tada:
- **Sep 25, 2024** : Released code for PALM
- **TO DO** : Released instructions for preparing datasets  


<br>

## Installation :gear:
1. Create a conda environment
```shell
conda create --name baple python=3.8
conda activate baple
```
2. Install PyTorch and other dependencies
```shell
git clone https://github.com/asif-hanif/palm
cd palm
pip install -r requirements.txt
```


</br>

## Model :white_square_button:
We have shown the efficacy of BAPLe on four medical foundation models: 

[MedCLIP](https://github.com/RyanWangZf/MedCLIP)&nbsp;&nbsp;&nbsp;[BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)&nbsp;&nbsp;&nbsp;[PLIP](https://github.com/PathologyFoundation/plip)&nbsp;&nbsp;&nbsp;[QuiltNet](https://quilt1m.github.io/)

Download the pre-trained models using the links provided below. Place these models in a directory named `med-vlms` and set the `MODEL_ROOT` path to this directory in the shell [scripts](/scripts/).



| Model | Link | Size |
|:-- |:-- | :-- |
| PENGI | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/EbiA2lv6mndHoAsEnZtv1F4BrCmmq9JZbT7FR6EZuCQ58A?e=5TvYr7) | 1.1 GB



Models should be organized according to the following directory structure:
```bash
med-vlms/
    ├── clip/
    ├── medclip/
    ├── biomedclip/ 
    ├── plip/
    ├── quiltnet/
 ```

## Datasets :page_with_curl:

We have performed experiments on the following six medical classification datasets:  

[COVID](https://arxiv.org/abs/2012.02238)&nbsp;&nbsp;&nbsp;[RSNA18](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)&nbsp;&nbsp;&nbsp;[MIMIC](https://arxiv.org/abs/1901.07042)&nbsp;&nbsp;&nbsp;[Kather](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002730)&nbsp;&nbsp;&nbsp;[PanNuke](https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2)&nbsp;&nbsp;&nbsp;[DigestPath](https://www.sciencedirect.com/science/article/pii/S1361841522001323)

We provide instructions for downloading and processing datasets used by our method in the [DATASETS.md](/datasets/DATASETS.md). 

| Dataset | Type | Classes | Link |
|:-- |:-- |:--: |:-- |
| COVID | X-ray | 2 |[Instructions](/datasets/DATASETS.md#covid) |
| RSNA18 | X-ray | 3 | [Instructions](/datasets/DATASETS.md#rsna18) |
| MIMIC | X-ray | 5 | [Instructions](/datasets/DATASETS.md#mimic) |
| Kather | Histopathology | 9 | [Instructions](/datasets/DATASETS.md#kather) |
| PanNuke | Histopathology | 2 | [Instructions](/datasets/DATASETS.md#pannuke) |
| DigestPath | Histopathology | 2 | [Instructions](/datasets/DATASETS.md#digestpath) |

</br>

All datasets should be placed in a directory named `med-datasets,` and the path of this directory should be specified in the variable `DATASET_ROOT` in the shell [scripts](/scripts/). The directory structure should be as follows:
```
med-datasets/
    ├── covid/
        |── images/
            |── train/
            |── test/
        |── classnames.txt
    ├── rsna18/
    ├── mimic/ 
    ├── kather/
    ├── pannuke/
    ├── digestpath/
 ```


Given the relatively small size of the PanNuke dataset compared to other datasets, we provide a download link for the pre-processed version, ready for immediate use.

| Dataset | Link | Size |
|:-- |:-- | :-- |
| PanNuke | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/Ed9DgWkCTf5JqbmMyRgNGTYBfMDrGQkNZwl_P3QSo8cj1Q?e=ZUM79g) | 531 MB |


</br>
<hr/>

## Code Structure :snowflake:
BAPLe code structure is borrowed from [COOP](https://github.com/KaiyangZhou/CoOp). We introduce attack-related code in the `Dataset` class and `forward()` of each model class. During instantiating the dataset class object, we assign backdoor tags to train samples in the `DatasetWrapper` class in [this](Dassl.pytorch/dassl/data/data_manager.py) file. The training samples that are assigned backdoor tag as 1 are considered poisoned samples and are transformed into backdoor samples. This transformation is done in the `forward()` of each model class. Code for these transformations is present in `trainers/backdoor.py` [file](trainers/backdoor.py). Model class for CLIP, PLIP, QuiltNet can be accessed [here](trainers/coop.py), for MedCLIP [here](trainers/coop_medclip.py) and for BioMedCLIP [here](trainers/coop_biomedclip.py). Prompt learning is managed `PromptLearner` class in each trainer file.

</br>

## Run Experiments :zap:

We have performed all experiments on `NVIDIA RTX A6000` GPU. Shell scripts to run experiments can be found in [scripts](/scripts/) folder. Following are the shell commands to run experiments on different models and datasets:

```shell
## General Command Structure
bash <SHELL_SCRIPT>   <MODEL_NAME>   <DATASET_NAME>   <CONFIG_FILE_NAME>   <NUM_SHOTS>
```

```shell
## MedCLIP
bash scripts/medclip.sh medclip covid medclip_ep50 32
bash scripts/medclip.sh medclip rsna18 medclip_ep50 32
bash scripts/medclip.sh medclip mimic medclip_ep50 32

## BioMedCLIP
bash scripts/biomedclip.sh biomedclip covid biomedclip_ep50 32
bash scripts/biomedclip.sh biomedclip rsna18 biomedclip_ep50 32
bash scripts/biomedclip.sh biomedclip mimic biomedclip_ep50 32


## PLIP
bash scripts/plip.sh plip kather plip_ep50 32
bash scripts/plip.sh plip pannuke plip_ep50 32
bash scripts/plip.sh plip digestpath plip_ep50 32


## QuiltNet
bash scripts/quiltnet.sh quiltnet kather quiltnet_ep50 32
bash scripts/quiltnet.sh quiltnet pannuke quiltnet_ep50 32
bash scripts/quiltnet.sh quiltnet digestpath quiltnet_ep50 32

```

Results are saved in `json` format in [results](/results/json) directory. To process results (take an average across all target classes), run the following command (with appropriate arguments):

```
python results/process_results.py --model <MODEL_NAME> --dataset <DATASET_NAME>
```

<details>
<summary>Examples</summary>

```shell
python results/process_results.py --model medclip --dataset covid
python results/process_results.py --model biomedclip --dataset covid
python results/process_results.py --model plip --dataset kather
python results/process_results.py --model quiltnet --dataset kather
```

</details>

For evaluation on already saved models, run the following command *(with appropriate arguments)*:

```shell
bash scripts/eval.sh   <MODEL_NAME>   <DATASET_NAME>   <CONFIG_FILE_NAME>   <NUM_SHOTS>
```

<details>
<summary>Examples</summary>

```shell
bash scripts/eval.sh medclip covid medclip_ep50 32
bash scripts/eval.sh biomedclip covid biomedclip_ep50 32
bash scripts/eval.sh plip kather plip_ep50 32
bash scripts/eval.sh quiltnet kather quiltnet_ep50 32
```

</details>


## Results :microscope:

![main figure](/media/table_1.png)
</br>
</br>
![main figure](/media/table_2.png)
</br>

## Citation :star:
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@article{hanif2024baple,
  title={BAPLe: Backdoor Attacks on Medical Foundational Models using Prompt Learning},
  author={Hanif, Asif and Shamshad, Fahad and Awais, Muhammad and Naseer, Muzammal and Khan, Fahad Shahbaz and Nandakumar, Karthik and Khan, Salman and Anwer, Rao Muhammad},
  journal={arXiv preprint arXiv:2408.07440},
  year={2024}
}
```
<!-- @inproceedings{hanif2024baple,
  title={BAPLE: Backdoor Attacks on Medical Foundational Models using Prompt Learning},
  author={Hanif, Asif and Shamshad, Fahad and Awais, Muhammad and Naseer, Muzammal and Khan, Fahad Shahbaz, Nandakumar, Karthick and Khan, Salman and Anwer, Rao Muhammad},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024},
  organization={Springer}
} -->
<hr/>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **asif.hanif@mbzuai.ac.ae**

<hr/>

## Acknowledgement :pray:
We used [COOP](https://github.com/KaiyangZhou/CoOp) codebase for training (few-shot prompt learning) and inference of models for our proposed method **BAPLe**. We thank the authors for releasing the codebase.

<hr />

