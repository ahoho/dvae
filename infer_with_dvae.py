import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from scipy import sparse

from dvae import data_iterator, CollapsedMultinomial, DVAE
from utils import load_sparse

_CUDA_AVAILABLE = torch.cuda.is_available()


def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def infer(model_dir_fpath, data): 
    """
    Loads the dvae model and gets the topic word distribution, then instantiates
    the encoder portion and does a forward pass to get the 
    """
    # get the topic word
    model_dir_fpath = Path(model_dir_fpath)
    device = torch.device("cuda") if _CUDA_AVAILABLE else torch.device("cpu")

    state_dict = torch.load(model_dir_fpath / "model.pt", map_location=device)
    beta = state_dict["params"]["decoder$$$eta_layer.weight"]
    topic_word = torch.transpose(beta, 0, 1).detach().numpy()

    # do a forward pass to get the document topics
    # first instantiate the model and load in the params
    config = load_yaml(model_dir_fpath / "config.yml")
    
    dvae = DVAE(
        vocab_size=topic_word.shape[1],
        num_topics=config["num_topics"],
        alpha_prior=config["alpha_prior"],
        embeddings_dim=config["encoder_embeddings_dim"],
        hidden_dim=config["encoder_hidden_dim"],
        dropout=config["dropout"],
        cuda=_CUDA_AVAILABLE,
    )
    dvae_dict = {
        k.replace("$$$", "."): v
        for k, v in state_dict['params'].items()
    }
    dvae.load_state_dict(dvae_dict, strict=False)
    dvae.eval()
    turn_off_bn = 1 * (config["epochs_to_anneal_bn"] > 0) # 0 means use BN, > 0 means no BN

    # then load the data for the forward pass
    batch_size = config["batch_size"]
    n = data.shape[0]
    train_batches = n // batch_size + 1

    # do the forward pass and collect outputs in an array
    doc_topic = np.zeros((n, config["num_topics"]), dtype=np.float32)
    losses = np.zeros(n, dtype=np.float32)
    for i, x_batch in tqdm(enumerate(data_iterator(data, batch_size, train_batches)), total=train_batches):
        x_batch = x_batch.to(device)
        doc_topic_batch = dvae.encoder(x_batch)
        doc_topic_batch = doc_topic_batch / doc_topic_batch.sum(1, keepdims=True)
        x_recon = dvae.decoder(doc_topic_batch, bn_annealing_factor=turn_off_bn)
        loss_batch = -CollapsedMultinomial(1, probs=x_recon).log_prob(x_batch)

        doc_topic[i * batch_size:(i + 1) * batch_size] = doc_topic_batch.detach().cpu().numpy().astype(np.float32)
        losses[i * batch_size:(i + 1) * batch_size] = loss_batch.detach().cpu().numpy().astype(np.float32)
    return topic_word, doc_topic, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_dir")
    parser.add_argument("--inference_data_file")
    parser.add_argument("--output_fpath")
    args = parser.parse_args()

    # load data
    x_eval = load_sparse(args.inference_data_file).astype(np.float32)
    _, doc_topics, _ = infer(args.model_save_dir, x_eval)
    Path(args.output_fpath).parent.mkdir(exist_ok=True, parents=True)
    np.save(args.output_fpath, doc_topics)

