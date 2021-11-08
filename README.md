# address-parser
This is a minimal implementation of a US address parser built using [spaCy NLP](https://spacy.io/usage/spacy-101) library. This [blog post](https://medium.com/globant/building-an-address-parser-with-spacy-e3376b7cff) covers the implementation and execution details at length.
<br><br>
# Prerequisites
- Python v3.x
- [spaCy v3.x](https://spacy.io/usage#installation)
<br><br>
# Folders 

## Training corpus
A sample corpus of US addresses to train/test the parser is present under [corpus/dataset](https://github.com/swapnil-saxena/address-parser/tree/main/corpus/dataset) folder.
JSON based rules required by Entity ruler are present under [corpus/rules]()

## Config
[config](https://github.com/swapnil-saxena/address-parser/tree/main/config) contains files for initializing training parameters:

**base_config.cfg**: Initializes pipeline and training batch size.
**base_config_er.cfg**: Similar as base_config but with additional entity ruler settings.
**config.cfg**: Pre filled config file obtained after executing inti fill-config.
**config_er.cfg**: Pre filled config file with additional entity ruler settings.

##  Output
[output](https://github.com/swapnil-saxena/address-parser/tree/main/output) contains final trained models (with and without entity rules)

<br><br>
# Training prerequisites
Before starting the training process, we need to: 

i) Obtain a pre filled training config which has the required training parameters.

ii) Build spacy-docbin (binary serialized representation) files for training and test dataset.

**Pre filled training config**:   Below command can be executed from command-line to get a pre filled config file. This would take as input the **base_config.cfg** file and churn out the pre filled training config file: **config.cfg**.
>python -m spacy init fill-config config\base_config.cfg config\config.cfg

Similarly, to get entity-ruler based config, pointing this command to the **base_config_er.cfg** would churn out the pre filled config : **config_er.cfg**

**Prepare spacy-docbins**: Finally, a spacy-docbin file can be obtained by executing [training_data_prep.py](https://github.com/swapnil-saxena/address-parser/blob/main/training_data_prep.py).
>python training_data_prep.py

This would take raw csv [training/test datasets](https://github.com/swapnil-saxena/address-parser/tree/main/corpus/dataset) as inputs and churn out docbin files under [corpus/spacy-docbins](https://github.com/swapnil-saxena/address-parser/tree/main/corpus/spacy-docbins) folder.

<br>

# Training loop execution

To start the training process, below train command can be executed:
>**python -m spacy train config\config.cfg --paths.train corpus\spacy-docbins\train.spacy --paths.dev corpus\spacy-docbins\test.spacy --output output\models --training.eval_frequency 10 --training.max_steps 300**

This saves the output NER models under [output](https://github.com/swapnil-saxena/address-parser/tree/main/output) folder.

<br><br>

# Predictions
Predictions for a few sample US addresses can be checked by executing [predict.py](https://github.com/swapnil-saxena/address-parser/blob/main/predict.py)
>python predict.py

Output:
>**Address string -> 130 W BOSE ST STE 100, PARK RIDGE, IL, 60068, USA  
>Parsed address -> [('130', 'BUILDING_NO'), ('W BOSE ST', 'STREET_NAME'), ('PARK RIDGE', 'CITY'), ('IL', 'STATE'), ('60068', 'ZIP_CODE'), ('USA', 'COUNTRY')]**
