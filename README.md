# Multimodal Complaint Detection and Aspect Identification

## Setup guide:

First, install the requirements:
```
pip install -r requirements.txt
```

Next, ensure that complaint.csv (or any file name, the main dataset) is present in root folder, and path is added preprocess_data.py.  
Then run:
```
python preprocess_data.py
```

After this, to generate the training instances, run:
```
python generate_training_examples.py
```  
Transcripts and audio embeddings will be generated in new_data.csv and audio_embeddings.csv in root folder respectively.  
Videos will be downloaded to downloads folder in root directory, and keyframes will be stored in the testframes folder.

## Running experiments:
To train and test the model (configure GPU accordingly)

```
cd models
python main.py
```