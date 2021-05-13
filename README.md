# Knowlege Base Extension with Text Enhanced OpenKI

This repository implements the OpenKI model [Zhang et.al.] and adds to that model text encoding 
capabilities [Wood et.al.]. T

The OpenKI model utilises a graph of predicate triples with entities linked to a knowledge base (KB), perhaps extracted 
with OpenIE tools, to infer new triples to add to the knowledge base. It does not attempt to map extracted predicates 
directly to KB relations, but instead looks at the joint KB-OpenIE graph (this makes sense if OpenIE entities are 
linked to KB entities) to make it's inferences. Please see the above papers for model details

# Installing
This code was written using Python 3.7. To install, clone this repository and run `pip install -r requirements.txt`, 
then run these commands to install RAdam:

    > git clone https://github.com/LiyuanLucasLiu/RAdam
    > cd RAdam
    > pip install -u .

# Preprocessed Data
Pre-processed data used in the above two papers is available from the following:

| Data Set  |  URL  |
| ---------------- | --- |
| OpenIE + ClueWeb | https://github.com/zhangdongxu/relation-inference-naacl19/blob/master/data_reverb.zip?raw=true |
| NYT              | https://drive.google.com/file/d/1_ayvXQt8dIfrafpAbOHRN8oqdzgdimjM/view?usp=sharing |

Unzip/untar these into the subfolder `data/reverb` and `data/nyt` for the examples below (any location will do &mdash; 
just change the  `--data-folder` option when invoking `main.py`). To download the google drive link, please see 
[this SO answer](https://stackoverflow.com/a/50670037/420867). 
See [Data Preprocessing Scripts](#data-preprocessing-scripts) below for instructions on how to pre-process your own data.

# Evaluating a Model
To run a trained model on evaluation and test data:

    python main.py --load-model <path-to-model.pt-file> --validate --test

The models that gave the results in the paper are available [here](http://to-be-completed/). 

# Training A Model
To train, for example, the ENE model without text enhancement on NYT data from the paper:

    python main.py  --label NYT-Ent_D-FastText --epochs 150 --optimizer RAdam --train --entity-nbhood-scoring \
     --data-folder data/nyt --data-source nyt --data-variants ignore_test_NA \
     --train-with-dev-preds

For the other models in the paper, change the following options:

| _Data Variant_      | Parameter Setting |
| -------------       | ----------------- |
| NYT                 | `--data-folder data/nyt --data-source nyt` |
| OpenIE + ClueWeb    | `--data-folder data/reverb --data-source reverb` |
| ___Text Encoding___   |  |
| FastText encoding   | `--word-embeddings fasttext --word-embed-file <path/to/crawl-300d-2M.vec>` |
| BERT (cls) encoding | `--word-embeddings bert --bert-pipeline feature-extraction --bert-use-cls` |
| BERT (avg) encoding | `--word-embeddings bert --bert-pipeline feature-extraction` |
| ___Included Texts___  |  |
| No included texts   | `--data-variants ignore_test_NA` |
| Entity             | `--entity-word-embeds-fc-tanh-sum --entity-static-embeds` | 
| +Entity Descriptions| `--data-vairants entity-text-with-descriptions ignore_test_NA` | 
| Pred/Rel           | `--predicate-word-embeds-concat-fc-tanh --pred-static-embeds` |  

To combine eg. entity and predicate/relation text encoding, simply combine the command line parameters. 
Note that the parameter `--data-variants` takes multiple options, and should always include `ignore_test_NA`
for evaluations presented in the paper. 
Program arguments can also be supplied in a text fie (spaces in argument list repalced with newlines) 
and loaded with the command line syntax `@<args_file_name>`.

During training, several output files are produced in the folder `output` (this can be overridden with the 
`--output-folder` command line argument). The output files are prefixed by the a string that combines the values 
provided to `--label` and `--jobid` : `f"{args.label}_{args.jobid}_"`. 
If this string contains the path delimeter ("/" on linux systems), the files will be created in a subfolder of the 
output folder. If `--jobid` is not specified, a time stamp will be used for jobid.

The files produced are:

    <base_name>.log
    <base_name>args.txt
    <base_name>_model_final.pt
    <base_name>_optimiser_final.pt

The `..._final.pt` files are updated at the end of each epoch or after a partially completed epoch if the model stops
during an epoch. This can happen due to a time limit (set with `--max-inference-hours`) or due to an exception. 

In addition, each time the evaluation score exceeds the best so far (evaluation is performed initially and after eacy 
epochs), the following files are produced:

    <base_name>BEST_<best-score>args.txt
    <base_name>_model_BEST_<best-score>.pt
    <base_name>_optimiser_BEST_<best-score>.pt

# Data Preprocessing Scripts
The folder `data_analysis/nyt-processing/` contains scripts for pre-processing data for use by `main.py`. 

<to be completed...>

# Citation

Please cite the following papers if you use this code base:

    Ian D. Wood, Stephen Wan, Mark Johnson
    Integrating Lexical Information into Entity Neighbourhood Representations for Relation Prediction
    NAACL 2021
    
    Zhang, Dongxu, Subhabrata Mukherjee, Colin Lockard, Luna Dong, and Andrew McCallum. 
    “OpenKI: Integrating Open Information Extraction and Knowledge Bases with Relation Inference.” 
    NAACL 2019
    https://doi.org/10.18653/v1/N19-1083.
