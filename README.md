**QA model for [Tensorflow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering)**

trained in 2 steps:
1. for each question one positive and one random negative samples were created, the model was trained using resulting dataset
2. inference model from step 1 to get predictions for all paragraphs, then 'hard' negatives were selected for new training dataset; continue training the model from step 1 using this new dataset

To reproduce dataset preprocessing, copy https://www.kaggle.com/c/tensorflow2-question-answering/data?select=simplified-nq-train.jsonl to `./data/` and run

    bash prepare_datasets.sh

To enable mixed precision training, download & install apex:

    git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext


Training using single GPU and mixed precision:

    bash run.sh

Distributed training on 4 GPUs using mixed precision:

    bash distrib_run.sh
