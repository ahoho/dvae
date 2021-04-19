import logging
from os import stat
import warnings
from typing import Dict, Optional, Union, Tuple
from pathlib import Path

import typer
import numpy as np
import pandas as pd
import tqdm
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO, JitTraceMeanField_ELBO
from pyro.optim import Adam

from utils import compute_npmi, compute_topic_overlap, compute_tu, load_sparse, load_json, save_topics

logger = logging.getLogger(__name__)

app = typer.Typer()


class L1RegularizedTraceMeanField_ELBO(TraceMeanField_ELBO):
    def __init__(self, *args, l1_params=None, l1_weight=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_params = l1_params
        self.l1_weight = l1_weight

    @staticmethod
    def l1_regularize(param_names, weight):
        params = torch.cat([pyro.param(p).view(-1) for p in param_names])
        return weight * torch.norm(params, 1)


    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss_standard = self.differentiable_loss(model, guide, *args, **kwargs)
        loss = loss_standard + self.l1_regularize(self.l1_params, self.l1_weight)

        loss.backward()
        loss = loss.item()

        pyro.util.warn_if_nan(loss, "loss")
        return loss


class CollapsedMultinomial(dist.Multinomial):
    """
    Equivalent to n separate `MultinomialProbs(probs, 1)`, where `self.log_prob` treats each
    element of `value` as an independent one-hot draw (instead of `MultinomialProbs(probs, n)`)
    """
    def log_prob(self, value: torch.tensor) -> torch.tensor:
        return ((self.probs + 1e-10).log() * value).sum(-1)


class Encoder(nn.Module):
    """
    Module that parameterizes the dirichlet distribution q(z|x)
    """
    def __init__(
            self,
            vocab_size: int,
            num_topics: int,
            embeddings_dim: int,
            hidden_dim: int,
            dropout: float,
        ):
        super().__init__()

        # setup linear transformations
        self.embedding_layer = nn.Linear(vocab_size, embeddings_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.second_hidden_layer = hidden_dim > 0
        if self.second_hidden_layer:
            self.fc = nn.Linear(embeddings_dim, hidden_dim)
            self.fc_drop = nn.Dropout(dropout)

        self.alpha_layer = nn.Linear(hidden_dim or embeddings_dim, num_topics)

        # this matches NVDM / TF implementation, which does not use scale
        self.alpha_bn_layer = nn.BatchNorm1d(
            num_topics, eps=0.001, momentum=0.001, affine=True
        )
        self.alpha_bn_layer.weight.data.copy_(torch.ones(num_topics))
        self.alpha_bn_layer.weight.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        embedded = F.relu(self.embedding_layer(x))
        embedded_do = self.embed_drop(embedded)

        hidden_do = embedded_do
        if self.second_hidden_layer:
            hidden = F.relu(self.fc(embedded_do))
            hidden_do = self.fc_drop(hidden)

        alpha = self.alpha_layer(hidden_do)
        alpha_bn = self.alpha_bn_layer(alpha)

        alpha_pos = torch.max(
            F.softplus(alpha_bn),
            torch.tensor(0.00001, device=alpha_bn.device)
        )

        return alpha_pos


class Decoder(nn.Module):
    """
    Module that parameterizes the obs likelihood p(x | z)
    """
    def __init__(
        self,
        vocab_size: int,
        num_topics: int, 
        bias_term: bool = True,
    ):
        super().__init__()

        self.eta_layer = nn.Linear(num_topics, vocab_size, bias=bias_term)

        # this matches NVDM / TF implementation, which does not use scale
        self.eta_bn_layer = nn.BatchNorm1d(
            vocab_size, eps=0.001, momentum=0.001, affine=True
        )
        self.eta_bn_layer.weight.data.copy_(torch.ones(vocab_size))
        self.eta_bn_layer.weight.requires_grad = False

    def forward(self, z: torch.tensor, bn_annealing_factor: float = 0.0) -> torch.tensor:
        eta = self.eta_layer(z)
        eta_bn = self.eta_bn_layer(eta)

        x_recon = (
            (bn_annealing_factor) * F.softmax(eta, dim=-1)
            + (1 - bn_annealing_factor) * F.softmax(eta_bn, dim=-1)
        )
        return x_recon
    
    @property
    def beta(self) -> np.ndarray:
        return self.eta_layer.weight.T.cpu().detach().numpy()


class DVAE(nn.Module):
    """
    Pytorch module for the Dirichlet-VAE
    """
    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        alpha_prior: float,
        embeddings_dim: int,
        hidden_dim: int,
        dropout: float,
        bias_term: bool = True,
        cuda: bool = True,
    ):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            embeddings_dim=embeddings_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            bias_term=bias_term,
        )

        if cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = cuda
        self.num_topics = num_topics
        self.alpha_prior = alpha_prior

    # define the model p(x|z)p(z)
    def model(
        self, x: torch.tensor,
        bn_annealing_factor: float = 1.0,
        kl_annealing_factor: float = 1.0
    ) -> torch.tensor:
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            alpha_0 = torch.ones(
                x.shape[0], self.num_topics, device=x.device
            ) * self.alpha_prior
            
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("doc_topics", dist.Dirichlet(alpha_0))
            # decode the latent code z
            x_recon = self.decoder(z, bn_annealing_factor)
            # score against actual data
            pyro.sample("obs", CollapsedMultinomial(1, probs=x_recon), obs=x)

            return x_recon

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(
        self, x: torch.tensor,
        bn_annealing_factor: float = 1.0,
        kl_annealing_factor: float = 1.0
    ):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z = self.encoder(x)
            # sample the latent code z
            with pyro.poutine.scale(None, kl_annealing_factor):
                pyro.sample("doc_topics", dist.Dirichlet(z))


def calculate_annealing_factor(
    current_batch: int,
    current_epoch: int,
    epochs_to_anneal: int,
    batches_per_epoch: int,
    min_af: float = 0.01,
) -> float:
    """
    Calculate annealing factor. Modified from pyro/examples/dmm.py
    """
    if epochs_to_anneal > 0:
        return min(
            1.0, # this hits when epochs_to_anneal > current_epoch
            min_af + (1.0 - min_af) * # when epochs_to_anneal <= current_epoch
            np.divide(
                (current_batch + current_epoch * batches_per_epoch + 1),
                (epochs_to_anneal * batches_per_epoch)
            )
        )
    else:
        return 0.0


def data_iterator(
    data: Union[np.ndarray, sparse.spmatrix],
    batch_size: int,
    num_batches: int
    ) -> torch.tensor:
    for i in range(num_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        if sparse.issparse(batch):
            batch = batch.toarray()
        yield torch.tensor(batch)


@app.command(help="Run a Dirichlet-VAE (with pytorch)")
def run_dvae(
        train_path: Path,
        output_dir: Path,
        eval_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
        num_topics: Optional[int] = None,
        to_dense: bool = True,

        encoder_embeddings_dim: int = 100,
        encoder_hidden_dim: int = 0, # setting to 0 turns off the second layer
        dropout: float = 0.25, # TODO: separate for enc/dec
        alpha_prior: float = 0.01,

        learning_rate: float = 0.001,
        topic_word_regularization: Optional[float] = None, 
        adam_beta_1: float = 0.9,
        adam_beta_2: float = 0.999,
        batch_size: int = 200,
        num_epochs: int = 200,
        epochs_to_anneal_bn: int = 0, # 0 is BN as usual; 1 turns BN off; >1 anneals from BN on to BN off
        epochs_to_anneal_kl: int = 100, # 0 throws error; 1 is standard KL; >1 anneals from 0 to 1 KL weight

        eval_words: int = 10,
        topic_words_to_save: int = 50,
        target_metric: str = "npmi",
        compute_eval_loss: bool = False,
        overlap_words: int = 10,
        max_acceptable_overlap: Optional[int] = None,
        eval_step: int = 1,
        
        seed: int = 42,
        gpu: bool = False,
        jit: bool = True,
    ) -> Tuple[DVAE, Dict[str, float]]:

    # clear param store
    pyro.clear_param_store()
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    pyro.enable_validation(__debug__)

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # load the data
    x_train = load_sparse(train_path).astype(np.float32)
    if to_dense:
        x_train = x_train.astype(np.float32).toarray()
    n_train = x_train.shape[0]

    vocab_size = x_train.shape[1]

    if eval_path is not None:
        x_val = load_sparse(eval_path).astype(np.float32)
        if not compute_eval_loss and not to_dense:
            x_val = x_val.to_csc() # slight speedup 
        if to_dense:
            x_val = x_val.toarray()
    else:
        compute_eval_loss = False # will be identical to train loss
        x_val, n_val = x_train, n_train

    n_val = x_val.shape[0]


    # load the vocabulary
    vocab = load_json(vocab_path) if vocab_path else None
    
    # setup the VAE
    vae = DVAE(
        vocab_size=vocab_size,
        num_topics=num_topics,
        alpha_prior=alpha_prior,
        embeddings_dim=encoder_embeddings_dim,
        hidden_dim=encoder_hidden_dim,
        dropout=dropout,
        bias_term=True,
        cuda=gpu,
    )

    # setup the optimizer
    adam_args = {
        "lr": learning_rate,
        "betas": (adam_beta_1, adam_beta_2), # from ProdLDA
    }
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    if topic_word_regularization:
        elbo = L1RegularizedTraceMeanField_ELBO(
            l1_params=["decoder$$$eta_layer.weight", "decoder$$$eta_layer.bias"],
            l1_weight=topic_word_regularization
        )
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = []
    results = []
    val_metrics = {
        "val_loss": np.inf,
        "npmi": 0,
        "tu": 0,
        "to": 0,
    }
    if max_acceptable_overlap is None:
        max_acceptable_overlap = float("inf")

    # training loop
    t = tqdm.trange(num_epochs, leave=True)
    train_batches = n_train // batch_size
    eval_batches = n_val // batch_size
    result_message = {}

    for epoch in t:
        # initialize loss accumulator
        random_idx = np.random.choice(n_train, size=n_train, replace=False)
        x_train = x_train[random_idx] #shuffle

        epoch_loss = 0.
        for i, x_batch in enumerate(data_iterator(x_train, batch_size, train_batches)):
            # if on GPU put mini-batch into CUDA memory
            if gpu:
                x_batch = x_batch.cuda()

            bn_af = calculate_annealing_factor(i, epoch, epochs_to_anneal_bn, train_batches)
            kl_af = calculate_annealing_factor(i, epoch, epochs_to_anneal_kl, train_batches)
            bn_af = torch.tensor(bn_af, device=x_batch.device)
            kl_af = torch.tensor(kl_af, device=x_batch.device)

            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x_batch, bn_af, kl_af)

        # report training diagnostics
        epoch_loss /= n_train
        train_elbo.append(epoch_loss)

        # evaluate on the validation set
        result_message.update({"tr loss": f"{epoch_loss:0.1f}"})

        if epoch % eval_step == 0:
            
            # get loss
            val_loss = epoch_loss # set to training loss if not being calculated
            if compute_eval_loss:
                val_loss = 0.
                for x_batch in data_iterator(x_val, batch_size, eval_batches):
                    if gpu:
                        x_batch = x_batch.cuda()
                    val_loss += svi.evaluate_loss(x_batch, bn_af, kl_af)
                val_loss /= n_val

            # get npmi
            beta = vae.decoder.beta

            # topic-uniqueness & topic overlap
            topic_terms = np.flip(beta.argsort(-1), -1)[:, :topic_words_to_save]
            curr_metrics =  {
                "val_loss": val_loss,
                "npmi": np.mean(compute_npmi(topic_terms, (x_val > 0), n=eval_words)),
                "tu": np.mean(compute_tu(topic_terms, n=eval_words)),
                "to": compute_topic_overlap(topic_terms, overlap_words, n=eval_words),
            }

            if val_metrics['to'] <= max_acceptable_overlap:
                val_metrics['val_loss'] = min(curr_metrics['val_loss'], val_metrics['val_loss'])
                val_metrics['npmi'] = max(curr_metrics['npmi'], val_metrics['npmi'])
                val_metrics['tu'] = max(curr_metrics['tu'], val_metrics['tu'])
                val_metrics['to'] = min(curr_metrics['to'], val_metrics['to'])

            if val_metrics[target_metric] == curr_metrics[target_metric]:
                pyro.get_param_store().save(Path(output_dir, "model.pt"))
                save_topics(topic_terms, vocab, Path(output_dir, "topics.txt"))

            result_message.update({k: v for k, v in curr_metrics.items()})
            results.append(curr_metrics)

        t.set_postfix(result_message)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(Path(output_dir, "results.csv"))
        print(
            f"Best NPMI: {results_df.npmi.max():0.4f} @ {np.argmax(results_df.npmi)}\n"
            f"Best TU @ this NPMI: {results_df.tu[np.argmax(results_df.npmi)]:0.4f}"
        )

    return vae, val_metrics


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=".*was not registered in the param store because requires_grad=False.*")
    warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

    app()
    #assert pyro.__version__.startswith('1.4.0')
    # parse command line arguments
    # parser = configargparse.ArgParser(
    #     description="parse args",
    #     config_file_parser_class=configargparse.YAMLConfigFileParser
    # )

    # parser.add("-c", "--config", is_config_file=True, default=None)
    # parser.add("--output_dir", required=True, default=None)
    # parser.add("--temp_model_dir", default=None, help="Temporary model storage during run, when I/O bound")


    # parser.add("--data_dir", default=None)
    # parser.add("-i", "--counts_fpath", default="train.npz")
    # parser.add("-v", "--vocab_fpath", default="train.vocab.json")
    # parser.add("-d", "--val_counts_fpath", default="dev.npz")
    # parser.add("--val_split", default=0.2, type=float)
    
    # parser.add("-k", "--num_topics", default=50, type=int)
    
    # parser.add("--encoder_hidden_dim", default=100, type=int)
    # parser.add("--encoder_dropout", default=0.2, type=float)
    # parser.add("--decoder_dropout", default=0.2, type=float)
    # parser.add("--alpha_prior", default=0.02, type=float)
    # parser.add("--pretrained_embeddings_dir", dest="pretrained_embeddings", default=None, help="directory containing vocab.json and vectors.npy")
    # parser.add("--update_embeddings", action="store_true", default=False)
    # parser.add("--second_hidden_layer", action="store_true", default=False)
    
    # parser.add('-lr', '--learning_rate', default=0.002, type=float)
    # parser.add("-b", "--batch_size", default=200, type=int)
    # parser.add("-n", '--num_epochs', default=101, type=int)
    # parser.add("--annealing_epochs", default=50, type=int)
    # parser.add("--minimum_annealing_factor", default=0.01, type=float)
    
    # parser.add("--eval_step", default=1, type=int)
    # parser.add("--val_metric_target", default="npmi", choices=["npmi", "loss", "tu"])
    # parser.add("--npmi_words", default=10, type=int)
    # parser.add("--tu_words", default=10, type=int)

    # parser.add("--run_seeds", default=[42], type=int, nargs="+", help="Seeds to use for each run")
    # parser.add('--cuda', action='store_true', default=False, help='whether to use cuda')
    # parser.add('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    # args = parser.parse_args()

    # # Filter two pyro warnings

    # # Run for each seed
    # base_output_dir = args.output_dir
    # Path(base_output_dir).mkdir(exist_ok=True, parents=True)

    # for i, seed in enumerate(args.run_seeds):
    #     # make subdirectories for each run
    #     args.seed = seed
    #     output_dir = Path(base_output_dir, str(seed))
    #     output_dir.mkdir(exist_ok=True, parents=True)
    #     args.output_dir = str(output_dir)
    
    #     # train
    #     print(f"\nOn run {i} of {len(args.run_seeds)}")
    #     model, metrics = main(args)
    
    # # Aggregate results
    # agg_run_results = []
    # for seed in args.run_seeds:
    #     output_dir = Path(base_output_dir, str(seed))
    #     results = pd.read_csv(Path(output_dir, "results.csv"))
    #     agg_run_results.append({
    #         "seed": seed,
    #         "best_npmi": np.max(results.val_npmi),
    #         "best_npmi_epoch": np.argmax(results.val_npmi),
    #         "best_tu_at_best_npmi": results.val_tu[np.argmax(results.val_npmi)],
    #         "best_tu": np.max(results.val_tu),
    #         "best_tu_epoch": np.argmax(results.val_tu),
    #         "best_npmi_at_best_tu": results.val_npmi[np.argmax(results.val_tu)],
    #     })

    # agg_run_results_df = pd.DataFrame.from_records(agg_run_results)
    # agg_run_results_df.to_csv(Path(base_output_dir, "run_results.csv"))
    # print(
    #     f"\n=== Results over {len(args.run_seeds)} runs ===\n"
    #     f"Mean NPMI: "
    #     f"{agg_run_results_df.best_npmi.mean():0.4f} ({agg_run_results_df.best_npmi.std():0.4f}) "
    #     f"@ epoch {np.mean(agg_run_results_df.best_npmi_epoch):0.1f} / {args.num_epochs}\n"
    #     f"Mean best TU @ best NPMI: {agg_run_results_df.best_tu_at_best_npmi.mean():0.4f}"
    # )