# Single and Multi-Task Gene Enhancer and Accessibility Prediction in Plant Genomics

## Introduction
Recent changes in global climate patterns (such as extreme average temperatures, droughts, and floods) have made the yields of Canadian farming methods much more unpredictable. Given these trends, scientists have taken to examining another major factor that can affect the success of a growing season: the genetics of the crop organism.

In the domain of genetics, certain segments of DNA can act as gene expression enhancers, giving rise to an increase in the number of RNA transcripts from that region which are eventually translated into proteins. Proteins are the building blocks of life and can have a significant effect on the cellular system they reside in, which can, in turn, lead to interesting downstream phenotypes in the host organism. Gene expression prediction is of particular interest to those in domains such as genetic engineering, which continues to be a key player in precision agriculture and smart food production systems. The accessibility of sequences within a genome are necessarily related to their enhancer activity.

The primary goal of this codebase is to train machine learning models that can accurately predict an increase or decrease in gene expression caused by the presence of a particular DNA sequence within Canadian crops. In this pursuit, we test a number of combinations of interesting deep learning architectures with different genetic enhancer strength and accessibility data sources.

## Function
Load in STARR-seq and/or ATAC omics datasets. Pre-process into machine-learning friendly format. Train models that use nucleotide frequency features as input (feed-forward neural network) or raw sequences as input (convolutional and recurrent neural networks) to predict enhancer activity or accessibility. Task can be formulated as regression or binary classification, and single or multi-target.

## Running Instructions
0. **Clone repository and navigate to the root directory.** 
1. **Unzip data file (stored on GDrive, contact for access) into `/data/raw`**. 

2. **Set up environment and initialize notebook**.
	1.1. If using conda and Jupyter, set up environment according to `versions.txt` and ignore the first cells in notebook that mount GDrive.
	1.2. If using GDrive and Colab, environment should already work with codebase. If not, consult `versions.txt`.

3.  **Run Code**.  Open `driver.ipynb`. Run cells to process and save raw datasets then train and evaluate models. 


## Notes
Change settings in `driver.ipynb` to change output datasets or model parameters. More information about available models can also be found here.

If you want to use GDrive and Colab, [Colab Pro](https://colab.research.google.com/signup) is required. This is because many datasets can't be processed and many models won't train with less than the 25GB RAM that Colab Pro allots. If you run a script and see that it seemingly terminated with `^C`, try again with `Runtime > Change Runtime Type > GPU Class > Premium`. Colab Pro+ is not necessary but wouldn't hurt.

An older, less focused and much more limited version of this codebase can be found here: https://github.com/sacha-davis/agronomics-project. 


## Credits
Model found in `/models` come from the MPRA-DragoNN publication:
`Movva R, Greenside P, Marinov GK, Nair S, Shrikumar A, Kundaje A (2019). Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays. PLoS ONE 14(6): e0218073.  [https://doi.org/10.1371/journal.pone.0218073](https://doi.org/10.1371/journal.pone.0218073)`