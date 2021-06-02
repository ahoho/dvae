from typing import Optional, Tuple, Callable

import numpy as np

import jax
from jax import random, lax, jit
import jax.numpy as jnp

from .utils import compute_npmi as _compute_npmi_numpy, load_sparse


def load_dataset(fpath, batch_size: Optional[int] = None, to_dense: bool = False) -> Tuple[Callable, Callable]:
    """
    Create dataset initialization and loading functions---only necessary for training
    data
    """
    dtm = load_sparse(fpath)
    num_records, vocab_size = dtm.shape
    idxs = np.arange(num_records)
    if not to_dense:
        # We will construct the batch from the indices in `get_batch`, which limits
        # memory use for large document-term matrices
        dtm = dtm.tocsr()
        max_toks = np.sum(dtm > 0, axis=1).max()
        row = jnp.tile(jnp.arange(batch_size), [max_toks, 1]).T
        col = np.zeros((num_records, max_toks), np.int32)
        data = np.zeros((num_records, max_toks), np.int32)
        for i in idxs:
            doc = dtm[i]
            toks = len(doc.indices)
            # Use leading zeros since they'll be overwritten during `index_update`
            # by the trailing data
            col[i][-toks:] = doc.indices
            data[i][-toks:] = doc.data
    else:
        dtm = jnp.array(dtm.toarray())

    if not batch_size:
        batch_size = num_records

    def init(key: Optional[jnp.ndarray] = None):
        idxs_shuffled = random.permutation(key, idxs) if key is not None else idxs
        data_dict = {}
        if not to_dense:
            data_dict["col"] = jnp.array(col[idxs_shuffled])
            data_dict["data"] = jnp.array(data[idxs_shuffled])

        data_dict['idxs'] = jnp.array(idxs_shuffled)
        return num_records // batch_size, data_dict

    def get_batch(i, data_dict):
        if not to_dense:
            # Construct the batch from the indices
            col = lax.dynamic_slice_in_dim(data_dict['col'], i * batch_size, batch_size)
            data = lax.dynamic_slice_in_dim(data_dict['data'], i * batch_size, batch_size)
            batch = jnp.zeros((batch_size, vocab_size))
            return jax.ops.index_update(batch, jax.ops.index[row, col], data, indices_are_sorted=True)
        else:
            ret_idx = lax.dynamic_slice_in_dim(data_dict['idxs'], i * batch_size, batch_size)
            return jnp.take(dtm, ret_idx, axis=0)

    return init, get_batch


def compute_npmi(sorted_topics, bin_ref_counts, n=10):
    if isinstance(bin_ref_counts, jnp.ndarray):
        return _compute_npmi_jax(sorted_topics, bin_ref_counts, n=n)
    else:
        return _compute_npmi_numpy(sorted_topics, bin_ref_counts, n=n)

@jax.partial(jit, static_argnums=(1,))
def extract_top_words(beta: jnp.ndarray, n: int = 10):
    num_topics, num_words = jnp.shape(beta)
    


@jax.partial(jit, static_argnums=(2,))
def _compute_npmi_jax(sorted_topics: jnp.ndarray, bin_ref_counts: jnp.ndarray, n: int = 10) -> jnp.ndarray:
    """jit-compatible version of compute_npmi"""
    num_topics, _ = jnp.shape(sorted_topics)
    num_docs, _ = jnp.shape(bin_ref_counts)

    sorted_topics = lax.dynamic_slice_in_dim(sorted_topics, 0, n, axis=-1)

    def body_fn_topics(k, npmi_means):
        """Loops over the k topics and fills in the `k`-length `npmi_means vector`"""
        indices = jnp.take(sorted_topics, k, axis=0)

        def body_fn_outer(i, npmi_vals_outer):
            """Outer loop, `i` from 0 to `n`"""

            def body_fn_inner(j, npmi_vals_inner):
                """Inner loop, `j` from `i+1` to `n`, calculating PMI for each (i, j)"""
                index1 = jnp.take(indices, i, axis=0)
                index2 = jnp.take(indices, j, axis=0)
                col1 = jnp.take(bin_ref_counts, index1, axis=1)
                col2 = jnp.take(bin_ref_counts, index2, axis=1)
                c1 = col1.sum()
                c2 = col2.sum()
                c12 = jnp.sum(col1 * col2)
                npmi = lax.cond(
                    c12 == 0,
                    lambda x: 0.,
                    lambda x: (
                        (jnp.log(num_docs) + jnp.log(c12) - jnp.log(c1) - jnp.log(c2)) 
                        / (jnp.log(num_docs) - jnp.log(c12))
                    ),
                    0.
                )
                prev_sum_inner, prev_count_inner = npmi_vals_inner
                return prev_sum_inner + npmi, prev_count_inner + 1
            # Run the inner index loop
            return lax.fori_loop(i + 1, n, body_fn_inner, init_val=npmi_vals_outer)

        # Run the outer index loop
        npmi_sum, count = lax.fori_loop(0, n, body_fn_outer, init_val=(0., 0))
        return jax.ops.index_update(npmi_means, k, npmi_sum / count)

    # Run the topic loop
    return lax.fori_loop(0, num_topics, body_fn_topics, init_val=jnp.zeros(num_topics))