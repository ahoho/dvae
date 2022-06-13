import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from tqdm import tqdm

from dvae import data_iterator, DVAE
from utils import load_sparse

_CUDA_AVAILABLE = torch.cuda.is_available()


def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def retrieve_estimates(model_dir, eval_data_fpath=None, **kwargs): 
    """
    Loads the dvae model then instantiates the encoder portion
    and does a forward pass to get the training-set document-topic estimates

    If `eval_data_fpath` is provided, will infer new document-topic estimates for the data
    and the topic-word estimates will __not__ be returned
    """
    model_dir = Path(model_dir)
    config = load_yaml(model_dir / "config.yml")
    device = torch.device("cuda") if _CUDA_AVAILABLE else torch.device("cpu")

    if eval_data_fpath is None:
        eval_data = load_sparse(Path(config["input_dir"], "train.dtm.npz")).astype(np.float32)
    else:
        eval_data = load_sparse(eval_data_fpath).astype(np.float32)
    
    state_dict = torch.load(model_dir / "model.pt", map_location=device)

    # do a forward pass to get the document topics
    # first instantiate the model and load in the params
    dvae = DVAE(
        vocab_size=state_dict["params"]["decoder$$$eta_layer.weight"].shape[0],
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

    # then load the data for the forward pass
    batch_size = config["batch_size"]
    doc_topic = retrieve_doc_topic(dvae, eval_data, device, batch_size)

    if eval_data_fpath is None:
        beta = state_dict["params"]["decoder$$$eta_layer.weight"]
        return torch.transpose(beta, 0, 1).detach().cpu().numpy(), doc_topic
    else:
        return doc_topic


def retrieve_doc_topic(dvae, data, device, batch_size=200):
    """
    Given a dvae model and data, do a forward pass of the encoder
    """
    n = data.shape[0]
    train_batches = n // batch_size + 1
    dvae.eval()

    # do the forward pass and collect outputs in an array
    doc_topic = None
    for i, x_batch in tqdm(enumerate(data_iterator(data, batch_size, train_batches)), total=train_batches):
        x_batch = x_batch.to(device)
        doc_topic_batch = dvae.encoder(x_batch)
        # this is the mean of the dirichlet given params
        doc_topic_batch = doc_topic_batch / doc_topic_batch.sum(1, keepdims=True)

        if doc_topic is None: # initialize
            doc_topic = np.zeros((n, doc_topic_batch.shape[1]), dtype=np.float32)
        doc_topic[i * batch_size:(i + 1) * batch_size] = doc_topic_batch.detach().cpu().numpy().astype(np.float32)

    return doc_topic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--inference_data_file")
    parser.add_argument("--output_fpath")
    args = parser.parse_args()

    assert Path(args.model_dir, "model.pt").exists(), f"Model does not exist at {args.model_dir}/model.py"
    
    if args.inference_data_file is not None:
        doc_topic = retrieve_estimates(args.model_dir, args.inference_data_file)
        Path(args.output_fpath).parent.mkdir(exist_ok=True, parents=True)
        np.save(args.output_fpath, doc_topic)
    else:
        topic_word, doc_topic = retrieve_estimates(args.model_dir, None)
        np.save(Path(args.model_dir, "beta.npy"), topic_word)
        np.save(Path(args.model_dir, "train.theta.npy"), doc_topic)
