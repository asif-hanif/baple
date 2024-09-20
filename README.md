# BAPLe: Backdoor Attacks on Medical Foundational Models using Prompt Learning (MICCAI'24)

> [**BAPLe: Backdoor Attacks on Medical
Foundational Models using Prompt Learning**](https://arxiv.org/pdf/2408.07440)<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), [Fahad Shamshad](https://scholar.google.com/citations?user=d7QL4wkAAAAJ), [Muhammad Awais](https://scholar.google.com/citations?hl=en&user=bA-9t1cAAAAJ),
[Muzammal Naseer](https://scholar.google.com/citations?hl=en&user=tM9xKA8AAAAJ), [Fahad Shahbaz Khan](https://scholar.google.com/citations?hl=en&user=zvaeYnUAAAAJ), <br> 
> [Karthik Nandakumar](https://scholar.google.com/citations?hl=en&user=2qx0RnEAAAAJ), [Salman Khan](https://scholar.google.com/citations?hl=en&user=M59O9lkAAAAJ) and,
[Rao Muhammad Anwer](https://scholar.google.com/citations?hl=en&user=_KlvMVoAAAAJ)


[![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/baple/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2408.07440)




<hr />

| ![main figure](/media/baple.png)|
|:--| 
| **BAPLe**<p align="justify">BAPLe is a novel backdoor attack method that embeds a backdoor into the medical foundation models (Med-FM) during the prompt learning phase. Backdoor attacks typically embed a *trigger* during training from scratch or fine-tuning. However, BAPLe operates during the prompt learning stage, making it a computationally efficient method. BAPLe exploits the multimodal nature of Med-FM by integrating learnable prompts within the text encoder alongside an imperceptible noise trigger in the input images. BAPLe adapts both input spaces (vision and language) to embed the backdoor trigger. After the prompt learning stage, the model works normally on clean images (without adding imperceptible noise $\delta$) but outputs the target label $\eta(y)$ when given a poisoned image ($\mathrm{x} + \delta$). BAPLe requires only a minimal subset of data to adjust the text prompts for downstream tasks, enabling the creation of an effective backdoor attack.</p> |

</br>
<hr />
</br>

| ![main figure](/media/intro.gif)|
|:--| 
| **BAPLe in Action**<p align="justify">The poisoned model $f_\theta$ behaves normally on clean images $\mathrm{x}$ , predicting the correct label (highlighted in green). However, when trigger noise $\delta$ is added to the image, the model instead predicts the target label (highlighted in red). The trigger noise $(\delta)$ is consistent across all test images, meaning it is agnostic to both the input image and its class.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Medical foundation models are gaining prominence in the medical community for their ability to derive general representations from extensive collections of medical image-text pairs. Recent research indicates that these models are susceptible to backdoor attacks, which allow them to classify clean images accurately but fail when specific triggers are introduced. However, traditional backdoor attacks necessitate a considerable amount of additional data to maliciously pre-train a model. This requirement is often impractical in medical imaging applications due to the usual scarcity of data. Inspired by the latest developments in learnable prompts, this work introduces a method to embed a backdoor into the medical foundation model during the prompt learning phase. By incorporating learnable prompts within the text encoder and introducing imperceptible learnable noise trigger to the input images, we exploit the full capabilities of the medical foundation models (Med-FM). Our method, BAPLe, requires only a minimal subset of data to adjust the noise trigger and the text prompts for downstream tasks, enabling the creation of an effective backdoor attack. Through extensive experiments with four medical foundation models, each pre-trained on different modalities and evaluated across six downstream datasets; we demonstrate the efficacy of our approach. BAPLe achieves a high backdoor success rate across all models and datasets, outperforming the baseline backdoor attack methods. Our work highlights the vulnerability of Med-FMs towards backdoor attacks and strives to promote the safe adoption of Med-FMs before their deployment in real-world applications. 
</i></p>

</br>
<hr />
</br>

<!-- 
## Backdoor Attack - Primer
<p align="justify">
A backdoor attack involves embedding a <i>visible/hidden</i> trigger (a small random or patterned patch) within a deep learning model during its training or fine-tuning phase. When the model encounters this trigger in the input data during inference, it produces a predefined output while performing normally on clean data.
</p> 

<p align="justify">
In a supervised classification task, a normally trained classifier $f_{\theta}: \mathcal{X} \rightarrow \mathcal{Y}$  maps a <i>clean</i> input image $\mathrm{x} \in \mathcal{X}$ to a label $y \in \mathcal{Y}$. Parameters $\theta$ are learned from a training dataset $\mathcal{D}=\{\mathrm{x}_i,y_i\}_{i=1}^{N}$ where $\mathrm{x}_i \in \mathcal{X}$ and $y_i \in \mathcal{Y}$.
</p> 

<p align="justify">
In a typical backdoor attack, the training dataset $\mathcal{D}$ is split into clean $\mathcal{D}_{c}$ and poison subsets $\mathcal{D}_{p}$, where $\vert\mathcal{D}_{p}\vert\ll N$. In $\mathcal{D}_p$, each sample $(\mathrm{x}, y)$ is transformed into a backdoor sample $(\mathcal{B}(x),\eta(y))$, where $\mathcal{B}: \mathcal{X} \rightarrow \mathcal{X}$ is the backdoor injection function and $\eta$ denotes the target label function. During the training/fine-tuning phase of backdoor attacks, the <i>victim</i> classifier $f_{\theta}$ is trained/fine-tuned on a mix of the clean dataset $\mathcal{D}_c$ and the poisoned dataset $\mathcal{D}_p$. The following objective functions are optimized to embed the backdoor in the model:
</p>


```math
\underset{ \theta }{\mathbf{minimize}}  \sum_{(\mathrm{x},y)\in\mathcal{D}_c} \lambda_c\cdot \mathcal{L}(f_{\theta}(\mathrm{x}), y) ~~+ \sum_{(\mathrm{x},y)\in\mathcal{D}_p} \lambda_p \cdot \mathcal{L}(f_{\theta}(\mathcal{B}(\mathrm{x})), \eta(y))
```

<p align="justify">
where $\mathcal{L}(\cdot)$ denotes the cross-entropy loss, and $\lambda_c$ and $\lambda_p$ are hyperparameters adjusting the balance of clean and poison data loss contributions.
</p>

<p align="justify">
After training, $f_{\theta}$ behaves similarly on clean input $\mathrm{x}$ as the original classifier (trained entirely on clean data), yet alters its prediction for the backdoor image $\mathcal{B}(\mathrm{x})$ to the target class $\eta(y)$, i.e.  $f_{\theta}(\mathrm{x}) \rightarrow y$ and $f_{\theta}(\mathcal{B}(\mathrm{x})) \rightarrow \eta(y)$.
</p>

## ZeroShot Inference in VLMs - Primer
<p align="justify">
ZeroShot inference in vision-language models (VLMs) refers to making predictions on new, unseen data without specific training. Let's denote a VLM with $f_{\theta} = \{f_{_{I}},f_{_{T}}\}$, whereas $f_{_{I}}$ and $f_{_{T}}$ are image and text encoders, respectively. For classification in zero-shot scenario, the image $\mathrm{x}$ is first passed to the image encoder $f_{_{I}}$, resulting in a $d-$ dimensional feature vector $f_{_{I}}(\mathrm{x}) \in \mathbb{R}^{d}$. Similarly, on the text encoder side, each class label $y_i \in \{\mathit{y}_{1}, \mathit{y}_{2}, \dots, \mathit{y}_{C} \}$ is wrapped within the class-specific text template, such as:
</p>

```math 
t_i = ``\mathrm{A~histopathology~image~of~\{CLASS~y_i\}}."
```
</br>

<p align="justify"> 
Each text prompt $(t_i)$ is fed to the text encoder $f_{_{T}}$, yielding text feature vector $f_{_{T}}(t_i) \in \mathbb{R}^{d}$. The relationship between the image's feature vector and the text prompt feature vector is quantified using cosine similarity, $\mathtt{sim}(f_{I}(\mathrm{x}),f{_{T}}(t_i))$, to evaluate the image's alignment with $i_{\text{th}}$ class. The class with the highest similarity score is selected as the predicted class label $\hat{y}$, i.e.
</p>

```math
\hat{y} = \underset{ i\in \{1,2,\dots,C\} }{\mathbf{argmax}} ~~~ \mathtt{sim}\big(f_{_{I}}(\mathrm{x})~,~f_{_{T}}(t_i)\big) 
```

## Prompt Learning
<p align="justify">
ZeroShot inference in VLMs requires hand-crafted text prompts for each class label. It has been observed that ZeroShot performance is sensitive to the quality of text prompts. <a href="https://arxiv.org/pdf/2307.12980">Prompt Learning</a> aims to learn these text prompts from the training data, avoiding the need for manual crafting. Many methods have been introduced for prompt learning for VLMs, but the first prominent method is <a href="https://github.com/KaiyangZhou/CoOp">COOP</a>, which learns the <i>context</i> of text prompts in the token-embedding space in few-shot setup. Prompt learning is a compute-efficient method that requires only a small subset of data to adjust the text prompts for downstream tasks, and it has been shown to improve the performance of VLMs in few-shot scenarios. 
</p>

## BAPLe
> <p align="justify">Prompt learning is a crucial component in our proposed method <b>BAPLe</b>. It employs a prompt learning setup that integrates a small set of learnable prompt token embeddings, $\mathcal{P}$, with class names, forming class-specific inputs $\mathrm{t}=\{t_1, t_2, \dots, t_C\}$ where $t_i = \{\mathcal{P}, y_i\}$. Denoting the model's prediction scores on clean image with $f_{\theta}(\mathrm{x})\in\mathbb{R}^{C}$:</p>

```math
f_{\theta}(\mathrm{x}) = \{~\mathtt{sim}(~f_{{I}}(\mathrm{x})~,~f{_{T}}(t_i)~)~\}_{i=1}^{C},
```
 
> where $\mathtt{sim}(\cdot)$ is cosine-similarity function. BAPLe optimizes the following objective function:

```math
\begin{gather} 
\underset{ \mathcal{P}~,~\delta }{\mathbf{minimize}}~~ \sum_{(\mathrm{x},y)\in\mathcal{D}_c} \lambda_c \cdot\mathcal{L}\big(f_{\theta}(\mathrm{x}),y\big) ~~+ \sum_{(\mathrm{x},y)\in\mathcal{D}_p} \lambda_p \cdot\mathcal{L}\big(f_{\theta}(\mathcal{B}(\mathrm{x})),\eta(y)\big),\nonumber \\
\mathbf{s.t.}~~~\|\delta\|_{{_{\infty}}} \le \epsilon,~~~~  \mathcal{B}(\mathrm{x}) = (\mathrm{x}+\delta)\oplus\mathrm{p}, \nonumber
\end{gather}
```

> <p align="justify">where $\delta$ represents the imperceptible backdoor trigger noise, $\epsilon$ is perturbation budget, $\mathrm{p}$ is the backdoor patch that can be a logo or symbol, $\mathcal{B}$ the backdoor injection function, and $\oplus$ represents an operation that combines the original image with the backdoor patch trigger. It must be noted that both vision and text encoders are kept in the frozen state. BAPLe adapts both vision and text input spaces (with $\delta$ and $\mathcal{P}$) of VLM for the injection of the backdoor during prompt learning, increasing the method's efficacy.  
</p>
-->

## Table of Contents
- [Installation](#installation)
- [Models](#models)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Experiments](#run-experiments)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

</br>
</br>

For more details, please refer to our [project web page](https://asif-hanif.github.io/baple/) or  [arxive paper](https://arxiv.org/pdf/2408.07440).

</br>
<hr/>


## Updates :rocket:
- **June 17, 2024** : Accepted in [MICCAI 2024](https://conferences.miccai.org/2024/en/) &nbsp;&nbsp; :confetti_ball: :tada:
- **Aug 12, 2024** : Released code for BAPLe
- **Aug 12, 2024** : Released pre-trained models (MedCLIP, BioMedCLIP, PLIP, QuiltNet) 
- **Aug 30, 2024** : Released instructions for preparing datasets (COVID, RSNA18, ~~MIMIC~~, Kather, PanNuke, DigestPath) 


<br>

## Installation :gear:
1. Create a conda environment
```shell
conda create --name baple python=3.8
conda activate baple
```
2. Install PyTorch and other dependencies
```shell
git clone https://github.com/asif-hanif/baple
cd baple
sh setup_env.sh
```

Our code uses [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch.git) codebase for dataset and training.

</br>

## Models :white_square_button:
We have shown the efficacy of BAPLe on four medical foundation models: 

[MedCLIP](https://github.com/RyanWangZf/MedCLIP)&nbsp;&nbsp;&nbsp;[BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)&nbsp;&nbsp;&nbsp;[PLIP](https://github.com/PathologyFoundation/plip)&nbsp;&nbsp;&nbsp;[QuiltNet](https://quilt1m.github.io/)

Download the pre-trained models using the links provided below. Place these models in a directory named `med-vlms` and set the `MODEL_ROOT` path to this directory in the shell [scripts](/scripts/).



| Model | Link | Size |
|:-- |:-- | :-- |
| CLIP | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/EbiA2lv6mndHoAsEnZtv1F4BrCmmq9JZbT7FR6EZuCQ58A?e=5TvYr7) | 1.1 GB
| MedCLIP | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/ET00tA0y5sxMo-tlAp3aMbsBUAZq0gOI1uviLy9dzdsbEw?e=bPTAUB) | 0.9 GB
| BioMedCLIP | - | -
| PLIP | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/ESv_3RFVSi1InR1UI53X43IBLeMgSeaGOA03dFkbnOe3wQ?e=m2K376) | 0.4 GB
| QuiltNet | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/EXlBHJFFOJZClKEQPtxyWTEBYRsBiMj9ZNjx08nK7qSzpA?e=nYfYrF) | 2.7 GB
| All-Models | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/EQBePYHzCK1PkFX76HxQGZABw3DigdV0Q9iGLcBgKtriyg?e=c3zgKf) | 5.0 GB


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

