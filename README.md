<p align="left">
<img src="assets/logo.png" alt="logo" width="200"/>
</p>

# Genetic Ancestry Prediction (GAP)

## Overview

GAP is a Python package, developed at the [Translational Genomics Lab](https://med.stanford.edu/genomics-of-diabetes.html) lead by Dr. Anna Gloyn at Stanford University, for predicting genetic ancestry from genotyping data using machine learning techniques. It provides tools for data preprocessing, model training, and evaluation to facilitate accurate ancestry inference. It shows superior performance compared to existing methods in ADMIXTURE and KING package when benchmarking against self-reported races from [Integrated Islet Distribution Program (IIDP)](https://iidp.coh.org/) and [The Human Pancreas Analysis Program (HPAP)](https://hpap.pmacs.upenn.edu/). 

GAP has been supported by both IIDP and the Stanford [Accelerate Innovation in Diabetes LeVeraging Unique PAthways iN Asians (ADVANCE)](https://asianhealth.stanford.edu/advance) Program.

## Pipeline
![](assets/pipeline.png)

## Installation

- using conda

```
git clone git@github.com:HaniceSun/gap.git
cd gap
conda env create -f environment.yml
conda activate gap
```

# Quick Start

```
input_vcf='INPUT.vcf.gz'

# using get-reference-data to download the 1000 Genomes reference data the first time you run GAP.
# It will take a while to download and process the 30+ GB data.
# skip this step if you have already downloaded the reference data and provide the paths as parameters:
# --reference ref/1000genomes_unrelated.vcf.gz when using merge-dataset-with-reference 
# --label_file ref/1000genomes_unrelated_sampleInfo.txt when using add-labels

gap get-reference-data --output_dir=ref

gap merge-dataset-with-reference --dataset $input_vcf --reference ref/1000genomes_unrelated.vcf.gz
gap feature-engineering

gap add-labels --label_file ref/1000genomes_unrelated_sampleInfo.txt
gap split-train-test --test_size 0.2

gap train-model --task Superpopulation
gap train-model --task Population --conditional true

gap eval-model --task Superpopulation
gap eval-model --task Population --conditional true

gap predict --task Superpopulation
gap predict --task Population --conditional true

gap summarize --conditional true

```

# Benchmark Results

- Superpopulation Prediction vs Self-reported Race in the IIDP cohorts

![](src/gap/data/GenetcicAncestry_vs_race_sankey.png)

## Citation

If you use GAP in your research, please cite the DOI: [10.5281/zenodo.18157870](https://doi.org/10.5281/zenodo.18157870)

## Author and License

**Author:** Han Sun

**Email:** hansun@stanford.edu

**License:** [MIT License](LICENSE)
