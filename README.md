# Decoupling Pragmatics: Discriminative Decoding for Referring Expression Generation

This is the code for our paper ["Decoupling Pragmatics: Discriminative Decoding for Referring Expression Generation" (ReInAct 2021)](https://aclanthology.org/2021.reinact-1.7/).

Steps for running the code:

1. Download the RefCOCO, RefCOCO+ and RefCOCOg datasets from [here](https://github.com/lichengunc/refer) and the COCO images from [here](https://cocodataset.org/#download)
2. Generate vocab and train models for RefCOCO, RefCOCO+ and RefCOCOg, put in respective folders in `data/model` (cf. model submodule; trained models & vocabs are provided below)
3. Decode the models using the different methods / parameters (cf. directory `run_decoding`; file paths have to be added to the bash files before running them. Generated expressions are provided)
4. Postprocess the generated expressions cf. directory `postprocess_generated`
    1. Clean from remaining `<end>` and `<unk>` tags (`clean_generated.py`)
    2. Sort out objects without distractors (`filter_single_object_imgs.py`)
    3. For informativity evaluation: Convert to format compatible with the MCN model (`convert_generated_to_mcn.py` for generated expressions, `convert_anns_to_mcn.py` for annotations)
5. Evaluate the generated expressions and place the results in `data/results`; cf. `evaluate` folder
    - Quality is evaluated using the [Referring Expression Datasets API](https://github.com/lichengunc/refer) (`refer` submodule)
    - Pragmatic informativity is evaluated using the MCN model (`MCN` submodule, [Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation; Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng and Rongrong Ji](https://arxiv.org/abs/2003.08813))
    - Diversity stats are calculated in `calculate_diversity.ipynb`; which is based on the [code](https://github.com/evanmiltenburg/MeasureDiversity) for this paper: [Measuring the Diversity of Automatic Image Descriptions; Emiel van Miltenburg, Desmond Elliott, Piek Vossen](https://aclanthology.org/C18-1147/). 

- For RSA, we rely on the code for this paper: [Pragmatically Informative Image Captioning with Character-Level Inference; Reuben Cohn-Gordon, Noah Goodman, Christopher Potts](https://aclanthology.org/N18-2070/); cf. `rsa` submodule.
- The code for ES decoding can be found in `decoding.py`, parts are based on [this](https://github.com/saiteja-talluri/Context-Aware-Image-Captioning) implementation.

## Data

Models and generated captions can be found [here](https://drive.google.com/drive/folders/1iom8qZPmJZKtf7E0renlyCKvzFbcpKf8?usp=sharing).