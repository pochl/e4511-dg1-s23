# Columbia E4511, Spring 2023, Project DG1: NLP for Legal Contract Review
This project adopts Question-Answering NLP model to perform clause extraction task, by following closely to the approach from the paper [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/pdf/2103.06268.pdf). Their [github](https://github.com/TheAtticusProject/cuad) page also provides us with the dataset and the fine-tune models. 

Since the CUAD dataset does not contain the labels for all clauses in our specific requirement, the aim of this project is to fine-tune the model on our own dataset and other public dataset that we can currate, as well as to experiment with other language models that could lead to better performance in legal text.

## Getting Start

### 1. Install
Clone repo and install requirements.txt in a Python>=3.7.0 environment. This project mainly utilises [Hugging Face](https://huggingface.co/) as the API for NLP model.
```
git clone https://github.com/pochl/contract-extraction
git cd contract-extraction
pip install -r requirements.txt
```

### 2. Create `resources` Folder
Create `resources` folder to store data, models, and other files. Choose your preferred method to create the following folder structure inside project's root directory

```
contract-extraction
    |
    |-- resources
            |-- data
            |-- models
            |-- tokenized_data
```
This can also be done via command line:
```
mk resources
mk resources/data
mk resources/models
mk resources/tokenized_data
```

### 3. Load CUAD data
Download CUAD [data](https://github.com/TheAtticusProject/cuad/raw/main/data.zip) from their github, extract the zip file, rename the data's root folder from "data" to "cuad" and place it in `resources/data` folder. 

Note that we can also use other datasets as long as they comply with SQuAD format.

### 4. Load Model
Since CUAD provides us with their fine-tuned model, you can either use their model directly or use models from others that are compatible with Hugging Face API. You can also use model from Hugging Face model hub as well, which required minor adjustment in `prepare_data.py` and `train.py`. To download CUAD's model, visit their official [webpage](https://zenodo.org/record/4599830#.Y_Pwq-zMKvA), download the preferred model, extract the content, and place it in `resources/models` folder.

### 5. Prepare Data
We have to prepare the raw data into tokenized data that can be processed by the training pipeline easily. This is done by running `prepare_data.py` script.
```
python prepare_data.py
```
Make sure `model_name` variable match the name of the model folder that has been downloaded.
The `prepare_data.py` script will tokenize the raw data and place it in `resources/tokenized_data` with the same name as the model's name. This is because different model has different tokenization mechanism and thus when using a new model, you need to generated tokenized data for such model. 

### 6. Fine-Tune Model
To fine-tune a model, run `train.py` script.
```
python train.py
```

Training configuration can be changed within the file.

## Key Points on How The Pipeline Works

### Prepare Contract
Put the NDA contracts you want to review under folder `File` (if not existing, create one).

### Predict and Annotate
run `predict.py` script
```
python predict.py
```

### View Annotated Contracts
The annotated contracts are placed under the folder `Annotated_Files`. \
After opening the contracts, paragraphs containing clauses that need to be reviewed and deleted are highlighted and commented. \
The comments include the category the clause belongs to and the clause itself. You can review it and decide if it need to be deleted. \
Currently, the model only deals with "Limitation of Liability", "Governing Law/Jurisdiction", "Non-competition", and "Non-solicitation".


