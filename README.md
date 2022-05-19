# Summarization as Indirect Supervision for Relation Extraction

Authors: Keming Lu, I-Hung Hsu, Wenxuan Zhou, Mingyu Derek Ma, Muhao Chen

## Overview

This repository provides codes of SuRE (Summarization as Relation Extraction). SuRE converts RE into a summarization formulation. SURE leads to more precise and resource-efficient RE based on indirect supervision from summarization tasks. To achieve this goal, we develop sentence and relation conversion techniques that essentially bridge the formulation of summarization and RE tasks. We also incorporate constraint decoding techniques with Trie scoring to further enhance summarization-based RE with robust inference. Experiments on three RE datasets demonstrate the effectiveness of SURE in both full-dataset and low-resource settings, showing that summarization is a promising source of indirect supervision to improve RE models.

## Key Idea of SuRE

The summarization task takes a context as the input sequence and a summary target is expected to be generated. To formulate RE as summarization, we first need to hint the summarization model which entity pair is targeted for summarization. To do so, we process the input sentence such that entity mentions and their type information will be highlighted. We explore existing entity marking tricks and also develop entity information verbalization technique that directly augments entity information as part of the context. The processed sentence will then be fed into SuRE. The summary targets for SuRE is created via verbalizing existing RE labels to templates, such as the Relation Verbalization subfigure in Fig. 1. In the training process, SuRE uses pretrained summarization models as a start point, and finetunes them with processed sentences as the input and verbalized relation descriptions as the targets. During inference, we incorporate several constrained inference techniques to help SURE decide the inferred relation

![Figure 1. Example for SuRE Inference](https://github.com/luka-group/SuRE/blob/main/figure1.png)

## Data
TBD

## Run

TBD

## Citing

TBD

## Requirements

SuRE is tested to work under Python 3.8, packages required include


All packages can be installed by `pip install -r requirements.txt`

## License

SuRE is licensed under the MIT License.
