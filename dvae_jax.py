from typing import Callable, Optional, Union
from pathlib import Path
import numpy

import typer
import tqdm
import pandas as pd
import numpy as np

from jax.tree_util import tree_flatten
from jax import jit, lax, random
from jax.experimental import stax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.nn.initializers import normal

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceMeanField_ELBO
from numpyro.optim import Adam

from utils_jax import compute_npmi, load_dataset
from utils import compute_topic_overlap, compute_tu, load_json, save_topics, load_sparse

app = typer.Typer()


def zeros_fixed(
    key: jnp.ndarray, shape: tuple, dtype: Union[str, jnp.dtype]=jnp.float32,
) -> jnp.ndarray:
    return lax.stop_gradient(jnp.zeros(shape, dtype))


class CollapsedMultinomial(dist.MultinomialProbs):
    """
    Equivalent to n separate `MultinomialProbs(probs, 1)`, where `self.log_prob` treats each
    element of `value` as an independent one-hot draw (instead of `MultinomialProbs(probs, n)`)
    """
    def log_prob(self, value):
        return (jnp.log(self.probs + 1e-10) * value).sum(-1)


def encoder(
    num_topics: int,
    embeddings_dim: int,
    hidden_dim: int,
    dropout: float,
) -> Callable:
    hidden = stax.Identity()
    if hidden_dim > 0:
        hidden = stax.serial(stax.Dense(hidden_dim), stax.Relu, stax.Dropout(dropout))
    return stax.serial(
        stax.Dense(embeddings_dim), stax.Relu,
        hidden,
        stax.Dropout(dropout),
        stax.Dense(num_topics),
        stax.BatchNorm(axis=1, epsilon=0.001, scale=False), stax.Softplus,
    )


def decoder(
    vocab_size: int,
    bias_term: bool = True,
) -> Callable:
    init_fn, apply_fn_partial = stax.serial(
        stax.Dense(vocab_size, b_init=normal() if bias_term else zeros_fixed()),
        stax.FanOut(2),
        stax.parallel(
            stax.Softmax,
            stax.serial(stax.BatchNorm(axis=1, epsilon=0.001, scale=False), stax.Softmax),
        )
    )
    def apply_fn(params, inputs, bn_annealing_factor=0.0, **kwargs):
        x, x_bn = apply_fn_partial(params, inputs, **kwargs)
        return (bn_annealing_factor) * x + (1 - bn_annealing_factor) * x_bn
    return init_fn, apply_fn


def guide(
    batch: jnp.ndarray,
    key: jnp.ndarray,
    bn_annealing_factor: float = None, # ignored in guide
    kl_annealing_factor: float = 1.0,
    num_topics: int = None,
    embeddings_dim: int = 100,
    hidden_dim: int = 0,
    alpha_prior: float = 0.02, # ignored in guide
    dropout: float = 0.0,
    bias_term: bool = True, # ignored in guide
) -> jnp.ndarray:
    batch_dim, vocab_size = jnp.shape(batch)
    encode = numpyro.module(
        name="encoder",
        nn=encoder(num_topics, embeddings_dim, hidden_dim, dropout),
        input_shape=(batch_dim, vocab_size),
    )
    z_alpha = lax.max(encode(batch, rng=key), 0.00001)
    with numpyro.handlers.scale(None, kl_annealing_factor):
        z = numpyro.sample("doc_topics", dist.Dirichlet(z_alpha))
    return z


def model(
    batch: jnp.ndarray,
    key: jnp.ndarray,
    bn_annealing_factor: float = 0.0, 
    kl_annealing_factor: float = None, # ignored in model
    num_topics: int = None,
    embeddings_dim: int = 100, # ignored in model
    hidden_dim: int = 100, # ignored in model
    alpha_prior: float = 0.02,
    dropout: float = 0.0, # ignored in model
    bias_term: bool = True,
) -> jnp.ndarray:
    batch_dim, vocab_size = jnp.shape(batch)
    decode = numpyro.module(
        name="decoder",
        nn=decoder(vocab_size, bias_term),
        input_shape=(batch_dim, num_topics),
    )
    alpha_0 = jnp.full((batch_dim, num_topics), alpha_prior)
    z_alpha_0 = numpyro.sample("doc_topics", dist.Dirichlet(alpha_0))
    x_recon = decode(z_alpha_0, rng=key, bn_annealing_factor=bn_annealing_factor)
    return numpyro.sample("obs", CollapsedMultinomial(probs=x_recon), obs=batch)


def calculate_annealing_factor(
    current_batch: int,
    current_epoch: int,
    epochs_to_anneal: int,
    batches_per_epoch: int,
    min_af: float = 0.01,
) -> float:
    """
    Calculate annealing factor. Modified from pyro/examples/dmm.py to be jit-able
    """
    return jnp.min(jnp.array([
        epochs_to_anneal, # since this is an int, only triggers if epochs_to_anneal == 0
        1.0, # this hits when epochs_to_anneal > current_epoch
        min_af + (1.0 - min_af) * # when epochs_to_anneal <= current_epoch
        jnp.divide(
            (current_batch + current_epoch * batches_per_epoch + 1),
            (epochs_to_anneal * batches_per_epoch)
        )
    ]))


@app.command(help="Run a Dirichlet-VAE")
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
):
    numpyro.set_platform("gpu" if gpu else "cpu")
    adam = Adam(learning_rate, betas=(adam_beta_1, adam_beta_2))
    svi = SVI(
        model=model,
        guide=guide,
        optim=adam,
        loss=TraceMeanField_ELBO(),
        num_topics=num_topics,
        embeddings_dim=encoder_embeddings_dim,
        hidden_dim=encoder_hidden_dim,
        alpha_prior=alpha_prior,
        dropout=dropout,
    )
    rng_key = PRNGKey(seed)
    train_init, train_fetch = load_dataset(train_path, batch_size, to_dense)
    vocab = load_json(vocab_path) if vocab_path else None

    if eval_path is not None:
        eval_data = (load_sparse(eval_path).tocsc() > 0) * 1
        if to_dense:
            eval_data = jnp.array(eval_data.toarray())
    else:
        train_data = (load_sparse(train_path).tocsc() > 0) * 1

    @jit
    def epoch_train(svi_state, epoch, train_data):
        def body_fn(i, val):
            loss_sum, svi_state = val
            rng_key_dropout = random.fold_in(rng_key, i)
            batch = train_fetch(i, train_data)
            bn_af = calculate_annealing_factor(i, epoch, epochs_to_anneal_bn, train_batches)
            kl_af = calculate_annealing_factor(i, epoch, epochs_to_anneal_kl, train_batches)

            svi_state, loss = svi.update(svi_state, batch, rng_key_dropout, bn_af, kl_af)
            loss_sum += loss
            return loss_sum, svi_state

        return lax.fori_loop(0, num_train, body_fn, (0., svi_state))

    if compute_eval_loss:
        raise NotImplementedError("Calculating the evaluation loss not yet implemented")

    num_train, train_data = train_init()
    sample_batch = train_fetch(0, train_data)
    rng_key, rng_key_init = random.split(rng_key, 2)
    svi_state = svi.init(rng_key_init, sample_batch, rng_key)

    results = []
    val_metrics = {
        "val_loss": np.inf,
        "npmi": 0,
        "tu": 0,
        "to": 0,
    }

    # training loop
    t = tqdm.trange(num_epochs, leave=True)
    train_batches = num_train // batch_size
    result_message = {}

    for epoch in t:
        rng_key, rng_key_data = random.split(rng_key, 2)
        _, train_data = train_init(rng_key_data)
        epoch_loss, svi_state = epoch_train(svi_state, epoch, train_data)

        # evaluate on the validation set -- TODO: make all this jittable
        result_message.update({"tr loss": f"{epoch_loss:0.1f}"})
        if epoch % eval_step == 0:
            beta = svi.get_params(svi_state)['decoder$params'][1][0]
            topic_terms = lax.dynamic_slice_in_dim(
                jnp.flip(beta.argsort(-1), -1), 0, topic_words_to_save, axis=1
            )
            npmi = jnp.mean(compute_npmi(topic_terms, eval_data, n=eval_words))
            topic_terms = topic_terms.tolist()
            curr_metrics =  {
                "npmi": npmi.item(),
                "tu": np.mean(compute_tu(topic_terms, n=eval_words)),
                "to": compute_topic_overlap(topic_terms, overlap_words, n=eval_words),
            }

            if val_metrics['to'] <= max_acceptable_overlap:
                val_metrics['val_loss'] = min(curr_metrics['val_loss'], val_metrics['val_loss'])
                val_metrics['npmi'] = max(curr_metrics['npmi'], val_metrics['npmi'])
                val_metrics['tu'] = max(curr_metrics['tu'], val_metrics['tu'])
                val_metrics['to'] = min(curr_metrics['to'], val_metrics['to'])

            if val_metrics[target_metric] == curr_metrics[target_metric]:
                svi.get_params(svi_state).save(Path(output_dir, "model.pt"))
                save_topics(topic_terms, vocab, Path(output_dir, "topics.txt"))

            result_message.update({k: v for k, v in curr_metrics.items()})
            results.append(curr_metrics)
            
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(Path(output_dir, "results.csv"))
            print(
                f"Best NPMI: {results_df.npmi.max():0.4f} @ {np.argmax(results_df.npmi)}\n"
                f"Best TU @ this NPMI: {results_df.tu[np.argmax(results_df.npmi)]:0.4f}"
            )

    return svi_state, val_metrics

if __name__ == "__main__":
    app()