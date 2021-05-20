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
just change the  `--data-folder` option when invoking `main.py`). 

To download the google drive link in a terminal, 
[gdown](https://pypi.org/project/gdown/) is a good tool (see 
[this SO answer](https://stackoverflow.com/a/50670037/420867) for tips on how to use it). 

[comment]: <> (See [Data Preprocessing Scripts]&#40;#data-preprocessing-scripts&#41; below for instructions on how to pre-process your own data.)

# Evaluating a Model
To run a trained model on evaluation and test data and calculate it's MAP (OpenIE + ClueWeb) resp. AUC-PR (NYT):

    python main.py --load-model <base-name-of-model> --run-to-load <model-run-identifier> --validate --test

Below are links to download the models that gave the best results in the paper (bolded results in Table 2). 
All these models use entity names with descriptions (no predicate texts) and FastText embeddings.  
See [Output Files](#output-files) below for further explanation of base names and run identifiers.

| Data Set        | Model  | `--load-model` | `--run-to-load` | URL |
| --------        | -----  | --- | --- | --- |
| OpenIE+ ClueWeb | ENE    | `OpenIE-ClueWeb-Ent-D/_47675615__47941299__47941300_` | `BEST_MAP_0.577957272529602_at_131` | https://drive.google.com/file/d/1luHXq2APHS2WKneCMPBJhBXWcke3_8vn/view?usp=sharing |
| OpenIE+ ClueWeb | OpenKI | `OpenIE-ClueWeb-Ent-D-OpenKI/_47781307__47857596__47857597_` | `BEST_MAP_0.5923588275909424_at_115` | https://drive.google.com/file/d/163N4UQj0cZ_t1PM7blBbFs56W2cQ3QND/view?usp=sharing |
| NYT             | ENE    | `NYT-Ent-D/_51588941__51588947__51588948_` | `BEST_AUC_0.9000850368617055_at_48` | https://drive.google.com/file/d/1YilESMRi6CH2I7w5pG3nAwz7ccwX9XLZ/view?usp=sharing |
| NYT             | OpenKI | `NYT-Ent-D-OpenKI/_51554475__51554495__51554496_` | `BEST_AUC_0.8909164465583739_at_43` | https://drive.google.com/file/d/1RGqmyfmQD61YFXLHVYYkOYtX6kjO2JkI/view?usp=sharing | 

To download the google drive link in a terminal,
[gdown](https://pypi.org/project/gdown/) is a good tool (see
[this SO answer](https://stackoverflow.com/a/50670037/420867) for tips on how to use it).

# Obtaining Scores From Triples
If you wish to obtain scores for candidate triples from a trained model, you can place the desired triples in a csv
file with columns (<subject_entity>, <relation>, <object_entity>), where each is given as a freebase id 
(e.g.: 'm.01nz1q6' for Yoko Ono or the relation 'base.activism.activist.area_of_activism'). 
Any entities or relations not present in the training data will result in zero scores. 

    python main.py --load-model <path-to-model.pt-file> --score-triples <path-to-triples-csv-file>

Alternatives, entities can be provided as integer indices in e2name.json (OpenIE + ClueWeb) resp. "id" field in
entities.json (NYT) with the added command line option `--score-triples-entity-by-index`. Similarly, relations can be 
provided as integer keys in p2name.json (OpenIE + ClueWeb) resp. row numbers in relations.csv (NYT) with the added 
command line option `--score-triples-relation-by-index`.

# Training A Model
To train, for example, the ENE model without text enhancement on NYT data from the paper:

    python main.py  --label NYT-Ent_D-FastText --epochs 150 --optimizer RAdam --train --entity-nbhood-scoring \
     --data-folder data/nyt --data-source nyt --data-variants ignore_test_NA \
     --train-with-dev-preds

For the other models in the paper, change the following options:

| Data Set        | Command Line Parameters |
| -------------       | ----------------- |
| NYT                 | `--data-folder data/nyt --data-source nyt` |
| OpenIE + ClueWeb    | `--data-folder data/reverb --data-source reverb` |

| Model Variant      |  |
| -------------       | ----------------- |
| Entity Neighbour Encoding (ENE) | `--entity-nbhood-scoring` |
| Dual Attention Scoring | `--entity-pair-dual-scoring` |
| "OpenKI" (ENE + dual attention) | `--entity-nbhood-scoring --entity-pair-dual-scoring` |

|    Text Encoding      |  |
| -------------       | ----------------- |
| FastText encoding   | `--word-embeddings fasttext --word-embed-file <path/to/crawl-300d-2M.vec>` |
| BERT (cls) encoding | `--word-embeddings bert --bert-pipeline feature-extraction --bert-use-cls` |
| BERT (avg) encoding | `--word-embeddings bert --bert-pipeline feature-extraction` |

|    Included Texts    |  |
| -------------        | ----------------- |
| All models           | `--data-variants ignore_test_NA` |
| Entity Names         | `--entity-word-embeds-fc-tanh-sum --entity-static-embeds` | 
| + Entity Descriptions | `--data-vairants entity-text-with-descriptions ignore_test_NA` | 
| Pred/Rel             | `--predicate-word-embeds-concat-fc-tanh --pred-static-embeds` |  

|     | Commonly Used Parameters |
| -------------       | ----------------- |
| `--label`           | Common label for multiple models. Use this to indicate parameter variations. |
| `--jobid`           | An id for this invocation of `main.py`. A date-stamp is used if omitted.     |
| `--train`           | Train the model. |
| `--epochs`          | (Max) number of epochs to train the model. |
| `--stop-on-no-best-for` | Stop if no validation improvement for this number of epochs. |
| `--test`            | Run the model on test data (after training if `--train` is set). |
| `--validate`        | Run the model on validation data (after training if `--train` is set). |
| `--data-folder`     | Folder containing pre-processed data |
| `--data-source`     | One of 'reverb' (for OpenIE + ClueWeb data) or 'nyt' (for NYT data) |
| `--embed-dim`       | Dimension of embeddings used for ENE representations |
| `--embed-dim-pairs` | Dimension of embeddings used for entity pair based representations (dual attention and query attention) |


To combine eg. entity and predicate/relation text encoding, simply combine the command line parameters. 
Note that the parameter `--data-variants` takes multiple options, should only appear once and shold always include 
`ignore_test_NA` for evaluations presented in the paper. 

Program arguments can also be supplied in a text fie (spaces in argument list repalced with newlines) 
and loaded with the command line syntax `@<args_file_name>`. 
There are many other options that can be supplied. `python main.py --help` lists all available options with extensive
descriptions of their function. Most were extensively tested in preliminary experiments and are best left at default
values. 

# Output Files
During training, several output files are produced in the folder `output` (this can be overridden with the 
`--output-folder` command line argument). The output files are prefixed by a `base_name` that combines the values 
provided to `--label` and `--jobid` : `base_name = f"{args.label}_{args.jobid}_"`. 
If this string contains the path delimeter ("/" on linux systems), the files will be created in a subfolder of the 
output folder. If `--jobid` is not specified, a time stamp will be used. 

When loading saved models, for `--load-model` use the `base_name` and for `--run-to-load` use either `final` 
(the default) or `BEST_<best-score-and-epoch>` as found in the output file name. 

The files produced are:

    <base_name>.log
    <base_name>args.txt
    <base_name>_model_final.pt
    <base_name>_optimiser_final.pt

The `..._final.pt` files are updated at the end of each epoch or after a partially completed epoch if the model stops
during an epoch. This can happen due to a time limit (set with `--max-inference-hours`) or due to an exception. 

In addition, each time the evaluation score exceeds the best so far (evaluation is performed initially and after each 
epoch), the following files are produced:

    <base_name>BEST_<best-score-and-epoch>args.txt
    <base_name>_model_BEST_<best-score-and-epoch>.pt
    <base_name>_optimiser_BEST_<best-score-and-epoch>.pt

The `...args.txt` files are valid json and contain all provided program arguments as well as some extra information such 
as the best evaluation score so far and the epoch at which it was attained. 

# Data Preprocessing Scripts
The folder `data_analysis/nyt-processing/` contains scripts for pre-processing data for use by `main.py`. These 
scripts were used to prepare the NYT data. The OpenIE + ClueWeb data was used as provided by the authors of the 
original OpenKI paper (see link above).  

We will shortly be providing variants of these scripts for use with user supplied data. 

# Citation

Please cite the following papers if you use this code base:

    Ian D. Wood, Stephen Wan, Mark Johnson
    Integrating Lexical Information into Entity Neighbourhood Representations for Relation Prediction
    NAACL 2021
    
    Zhang, Dongxu, Subhabrata Mukherjee, Colin Lockard, Luna Dong, and Andrew McCallum. 
    “OpenKI: Integrating Open Information Extraction and Knowledge Bases with Relation Inference.” 
    NAACL 2019
    https://doi.org/10.18653/v1/N19-1083.
