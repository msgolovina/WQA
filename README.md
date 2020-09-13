**Tensorflow 2.0 Question Answering**

trained using 8 x Tesla V100-SXM2-32GB (20 min on one epoch)

    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html


To reproduce dataset preprocessing, copy https://www.kaggle.com/c/tensorflow2-question-answering/data?select=simplified-nq-train.jsonl to `./data/` and run

    bash prepare_datasets.sh

To enable mixed precision training, download & install apex:

    git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext


Training using single GPU and mixed precision:

    bash run.sh

Distributed training on 4 GPUs using mixed precision:

    bash distrib_run.sh
