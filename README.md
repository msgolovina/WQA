**Tensorflow 2.0 Question Answering**

To replicate dataset preprocessing, copy https://www.kaggle.com/c/tensorflow2-question-answering/data?select=simplified-nq-train.jsonl to `./data/` and run

    bash prepare_datasets.sh

To enable mixed precision training, download & install apex:

    git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext


Training using single GPU and mixed precision:

    bash run.sh

Distributed training on 4 GPUs using mixed precision:

    bash distrib_run.sh
