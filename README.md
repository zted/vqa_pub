#Visual Question Answering

A simple LSTM and CNN model to run on the [VQA](http://visualqa.org) dataset. CNN is precomputed using VGG 19. One hot encoding and no embeddings used. More complex models coming soon!

Currently the model trains on train and validates on val, so use the appropriate splits when doing preprocessing

##Current numbers (no hyperparameter tuning done)

| Model    | val                 |
| ---------|:-------------------:|
| LSTM+CNN | 52.15% (Open Ended) |

##Before you run:
1. Follow the preprocessing steps from [here](https://github.com/VT-vision-lab/VQA_LSTM_CNN)
2. Install all dependencies, here we just need keras to run
3. Put all the relevant files (annotations, questions, processed image features, other features) in one directory
4. Change the config file suitable to that of your environment
