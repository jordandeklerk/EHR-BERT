import argparse
import logging
import os
import random
import wandb

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.EHRBert.bert_pretrain import BERT_Pretrain
from src.EHRBert.bert_config import BertConfig
from src.ehr_dataset import train_dataloader, eval_dataloader, train_dataset, tokenizer
from src.utils import metric_report, get_n_params

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_device_and_logging(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    return device

def load_model(args, tokenizer):
    logger.info('Loading Model: ' + args.model_name)
    if args.use_pretrain:
        logger.info("Using pretraining model")
        model = BERT_Pretrain.from_pretrained(args.pretrain_dir, dx_voc=tokenizer.dx_voc, proc_voc=tokenizer.proc_voc)
    else:
        config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
        model = BERT_Pretrain(config, tokenizer.dx_voc, tokenizer.proc_voc)
    logger.info('# of model parameters: %d', get_n_params(model))
    return model

def save_model(model, args, output_model_file):
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), output_model_file)
    with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
        fout.write(model.config.to_json_string())

def log_metrics(metrics, global_step, args, prefix='eval'):
    for k, v in metrics.items():
        logger.info(f'{prefix}/{k} at step {global_step}: {v}')
    if args.use_wandb:
        wandb.log({f'{prefix}/{k}': v for k, v in metrics.items()}, step=global_step)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='Bert-Pretraining', type=str, help="model name")
    parser.add_argument("--data_dir", default='./data', type=str, help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='./data', type=str, help="Pretraining model dir.")
    parser.add_argument("--train_file", default='data-comb-visit.pkl', type=str, help="Training data file.")
    parser.add_argument("--output_dir", default='./data/Bert-Pretraining', type=str, help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--use_pretrain", default=False, action='store_true', help="Use pretraining model")
    parser.add_argument("--threshold", default=0.3, type=float, help="Threshold")
    parser.add_argument("--max_seq_length", default=55, type=int, help="Max sequence length after tokenization")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training")
    parser.add_argument("--do_eval", default=True, action='store_true', help="Whether to run on the dev set")
    parser.add_argument("--do_test", default=False, action='store_true', help="Whether to run on the test set")
    parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="Initial learning rate for Adam")
    parser.add_argument("--weight_decay", default=1e-1, type=float, help="Weight decay for Adam optimizer")
    parser.add_argument("--num_train_epochs", default=15, type=int, help="Total number of training epochs")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--seed", type=int, default=1315, help="Random seed for initialization")
    parser.add_argument("--use_wandb", action='store_true', help="Use wandb for logging")
    parser.add_argument("--wandb_api_key", type=str, help="Wandb API key")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)

    seed_everything(args.seed)

    device = setup_device_and_logging(args)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    model = load_model(args, tokenizer)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    if args.do_train:
        # Initialize Wandb if specified
        if args.use_wandb:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            wandb.init(project="EHRBert-MTL", name=args.model_name, config=args)
        print('-'*100)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.batch_size)

        best_acc = 0
        global_step = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            total_loss = 0
            train_progress_bar = tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}/{args.num_train_epochs}")

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, dx_labels, proc_labels = batch

                loss, dx2dx, _, _, proc2proc = model(input_ids, dx_labels, proc_labels)
                loss.backward()

                total_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                train_progress_bar.update(1)

            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}, Train Loss: {avg_train_loss:.4f}")
            if args.use_wandb:
                wandb.log({"train/loss": avg_train_loss, "epoch": epoch}, step=global_step)

            train_progress_bar.close()

            if args.do_eval:
                model.eval()
                eval_loss = 0
                dx2dx_preds, dx_trues = [], []
                proc2proc_preds, proc_trues = [], []
                eval_progress_bar = tqdm(total=len(eval_dataloader), desc=f"Evaluating Epoch {epoch + 1}/{args.num_train_epochs}")

                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, dx_labels, proc_labels = batch
                    with torch.no_grad():
                        loss, dx2dx, _, _, proc2proc = model(input_ids, dx_labels, proc_labels)
                        eval_loss += loss.item()

                        dx2dx_preds.append(dx2dx.cpu().numpy())
                        proc2proc_preds.append(proc2proc.cpu().numpy())
                        dx_trues.append(dx_labels.cpu().numpy())
                        proc_trues.append(proc_labels.cpu().numpy())

                        eval_progress_bar.update(1)

                avg_eval_loss = eval_loss / len(eval_dataloader)
                dx2dx_metrics = metric_report(np.concatenate(dx2dx_preds, axis=0), np.concatenate(dx_trues, axis=0), args.threshold)
                proc2proc_metrics = metric_report(np.concatenate(proc2proc_preds, axis=0), np.concatenate(proc_trues, axis=0), args.threshold)

                eval_progress_bar.set_postfix({
                    "Train Loss": avg_train_loss,
                    "Eval Loss": avg_eval_loss,
                    "dx2dx_acc": dx2dx_metrics['prauc'],
                    "proc2proc_acc": proc2proc_metrics['prauc'],
                })

                log_metrics(dx2dx_metrics, global_step, args, prefix='eval_dx2dx')
                log_metrics(proc2proc_metrics, global_step, args, prefix='eval_proc2proc')

                if args.use_wandb:
                    wandb.log({"eval/loss": avg_eval_loss, "epoch": epoch}, step=global_step)

                logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}, Eval Loss: {avg_eval_loss:.4f}, dx2dx_acc: {dx2dx_metrics['prauc']:.4f}, proc2proc_acc: {proc2proc_metrics['prauc']:.4f}")

                if dx2dx_metrics['prauc'] > best_acc:
                    best_acc = dx2dx_metrics['prauc']
                    save_model(model, args, os.path.join(args.output_dir, "pytorch_model.bin"))

            eval_progress_bar.close()
            print("-"*100)

if __name__ == "__main__":
    main()