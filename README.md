# seq2seqNER
Sequence-to-Sequence Model for progressively learning new Named Entities in the text.
The model is implemented with PyTorch in Python3. 

## Model
The sequence-to-Sequence model consist of an encoder (Bidirectional LSTM) and a decoder (LSTM) with attention machenism implemented.

## Data Format
Existing models are trained on CoNLL 2003 NER dataset.
Other dataset can be used for training in the following format:

```
sentence_sequence<TAB>label_sequence
```

For example:
```
All passengers freed from Sudanese hijack plane .	O O O O S-MISC O O O
```

## Usage
For training:
```
python3 main.py --parameters_filepath=./parameters-train.ini 
```
Hyperparameters are set in parameters-train.ini
