
import logging
import warnings
from pathlib import Path
from typing import Optional, List

import typer

try:
    from .dvae_jax import run_dvae
except ImportError:
    from .dvae import run_dvae

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.command("Run a Dirichlet-VAE")
def run(
        input_dir: Path = typer.Argument(
            ...,
            exists=True,
            help=(
                "Directory containing training and evaluation data, filenames are "
                "optionally specified with `--train-fname` and `--eval-fname`"
            ),
        ),
        output_dir: Path = typer.Argument(
            ...,
            exists=True,
            help="Directory containing training and evaluation data",
        ),
        train_fname: Optional[str] = "train.dtm.npz",
        eval_fname: Optional[str] = "val.dtm.npz",
        vocab_fname: Optional[str] = "vocab.json",
        num_topics: Optional[int] = 20,

        encoder_embeddings_dim: int = 100,
        encoder_hidden_dim: int = 100,
        dropout: float = 0.2, # TODO: separate for enc/dec
        alpha_prior: float = 0.02,

        learning_rate: float = 0.001,
        batch_size: int = 200,
        num_epochs: int = 200,
        epochs_to_anneal: int = 50,
        eval_words: int = 10,
        to_dense: bool = True,
        
        seeds: List[int] = [11235],
        gpu: bool = False,
        pytorch_jit: bool = True,
    ):
        base_output_dir = output_dir
        Path(base_output_dir).mkdir(exist_ok=True, parents=True)
        # topic modeling is usually quick to run and estimates can sometimes be unstable;
        # seed directories for estimates are necessary
        for i, seed in enumerate(seeds):
            output_dir = Path(base_output_dir, str(seed))
            output_dir.mkdir(exist_ok=True)

            model, metrics = run_dvae(
                input_dir=input_dir,
                output_dir=output_dir,
                train_fname=train_fname,

            )



if __name__ == "__main__":
    # Filter two pyro warnings
    warnings.filterwarnings("ignore", message=".*was not registered in the param store because requires_grad=False.*")
    warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")
    app()