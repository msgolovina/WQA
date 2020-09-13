from dataset import NQIterableTestDataset
from models import BertForQA

import argparse
import logging
import numpy as np
from torch.utils.data import DataLoader
# from transformers import BertConfig, BertTokenizer
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--local-rank",
    type=int,
    default=-1,
    help="local_rank for distributed training on gpus",
)
parser.add_argument(
    "--test-path",
    type=str,
    default='data/toy_test.jsonl',
)
parser.add_argument(
    "--model-checkpoint-path",
    type=str,
    default='logs/test_run/output/checkpoint-10000/',
)
parser.add_argument("--no_cuda", action="store_true",
                    help="Whether not to use CUDA when available")

args = parser.parse_args()

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

logger.info('Loading model...')
if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

# bert_config = BertConfig.from_pretrained(config['pretrained_bert_name'])
# bert_config.num_labels = 5
model = BertForQA.from_pretrained(args.model_checkpoint_path)
if args.local_rank == 0:
    torch.distributed.barrier()
model.to(args.device)

batch_size = 1 * max(1, args.n_gpu)
dataset = NQIterableTestDataset(args.test_path)
dataloader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      pin_memory=True)

# multi-gpu evaluate
if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

# Eval!
logger.info("***** Running evaluation *****")
logger.info("  Batch size = %d", batch_size)
answers = []
pred_answers = []
top3_pred_answers = []

for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            batch = [torch.tensor(t) for t in batch]
            batch = [t.to(args.device) for t in batch]
            #batch_input_ids = batch[0]
            #batch_attention_mask = batch[1]
            #batch_token_type_ids = batch[2]
            #batch_y_start = batch[3]
            #batch_y_end = batch[4]
            #batch_y = batch[5]

            class_label = np.max(batch[-1])
            answer = np.argmax(batch[-1])

            la_scores = []
            for idx in len(batch[-1]):
                one_p = [x[idx] for x in batch]
                input_ids = one_p[0]
                attention_mask = one_p[1]
                token_type_ids = one_p[2]
                y_start = one_p[3]
                y_end = one_p[4]
                y = one_p[5]
                logits_start, logits_end, logits_class = model(
                    input_ids, attention_mask, token_type_ids
                )
                no_answer_score = logits_class[0]
                la_scores.append(1 - no_answer_score)

            predicted_answer = np.argmax(la_scores)
            top3 = np.argpartition(-logits_class, 3)[:3]
            top3_pred_answers.append(top3)

# calc results

def acc_at_3(y_true, top_3_pred):
    accuracy = []
    for y, pred in zip(y_true, top_3_pred):
        # y_true = np.array(y_true)
        # top_3_pred = np.array(top_3_pred)
        sample_accuracy = float(y in pred)
        accuracy.append(sample_accuracy)

    return np.mean(accuracy)


print('Accuracy: ', accuracy_score(answers, pred_answers))
print('Accuracy@3: ', acc_at_3(answers, pred_answers))
