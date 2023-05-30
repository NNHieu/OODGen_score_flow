
import gc
import io
import os
import time
from typing import Any

import flax
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from jax.random import KeyArray
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
# import wandb
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import bound_likelihood
import sde_lib
from absl import flags

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
# Keep the import below for registering all model definitions
from models import ncsnpp



from configs.subvp.svhn_ddpmpp_continuous import get_config

# Convert old checkpoint to compatible with new version lib
def convert_checkpoint(state):
  params_ema = state.params_ema.unfreeze()
  cond_tree = jtu.tree_map(lambda x: False, params_ema)
  def walk_dict(d,depth=0):
    for k,v in sorted(d.items(),key=lambda x: x[0]):
      if isinstance(v, dict):
        if 'GroupNorm' in k:
          for sk,sv in v.items():
            v[sk] = True
        else:
          walk_dict(v,depth+1)
  walk_dict(cond_tree)
  params_ema = jtu.tree_map(lambda p, cond: p.squeeze() if cond else p, params_ema, cond_tree)
  # object.__setattr__(state, 'params_ema', flax.core.frozen_dict.freeze(params_ema))
  state = state.replace(params_ema=flax.core.frozen_dict.freeze(params_ema))
  return state

# A data class for storing intermediate results to resume evaluation after pre-emption
@flax.struct.dataclass
class EvalMeta:
  ckpt_id: int
  bpd_round_id: int
  rng: Any


def restore_eval_meta(eval_dir, num_bpd_rounds, begin_ckpt, rng: KeyArray) -> EvalMeta:
  eval_meta = EvalMeta(ckpt_id=begin_ckpt, bpd_round_id=-1, rng=rng)
  try:
    eval_meta = checkpoints.restore_checkpoint(
      eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")
  except:
    logging.info("Unable to restore lastest eval meta checkpoint")
  if eval_meta.bpd_round_id < num_bpd_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = eval_meta.bpd_round_id + 1
  else:
    begin_ckpt = eval_meta.ckpt_id + 1
    begin_bpd_round = 0
  eval_meta = eval_meta.replace(ckpt_id=begin_ckpt, bpd_round_id=begin_bpd_round)
  return eval_meta

def save_eval_meta(eval_dir, num_bpd_rounds, eval_meta, ckpt, bpd_round_id, rng: KeyArray):
  eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
  # Save intermediate states to resume evaluation after pre-emption
  checkpoints.save_checkpoint(
    eval_dir,
    eval_meta,
    step = ckpt * num_bpd_rounds + bpd_round_id,
    keep=1,
    prefix=f"meta_{jax.host_id()}_", overwrite=True)

def randomize_tiles_shuffle_blocks(a, M, N, key):    
    m,n,p = a.shape
    b = a.reshape(m//M,M,n//N,N, p).swapaxes(1,2).reshape(-1,M*N, p)
    b = jax.random.permutation(key, b)
    return b.reshape(m//M,n//N,M,N,p).swapaxes(1,2).reshape(a.shape)

randomize_tiles_shuffle_blocks_batch = jax.vmap(randomize_tiles_shuffle_blocks, (0, None, None, 0))

def evaluate_bpd(
      config,
      workdir,
      eval_folder="eval",
      deq_folder="flowpp_dequantizer",
      block_shuffle=0):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  checkpoint_dir = os.path.join(workdir, "checkpoints")

  rng = jax.random.PRNGKey(config.seed + 1)

  # Setup SDEs
  sde = sde_lib.get_sde(config.training.sde, config.model)
  sampling_eps = config.sampling.smallest_time

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  # optimizer = losses.get_optimizer(config).create(initial_params)
  # optimizer = None
  state = mutils.OldState(step=0, optimizer=None, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  if config.eval.dequantizer:
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        additional_dim=None,
                                                        uniform_dequantization=False,
                                                        evaluation=True)  # For data-dependent initialization. Must take values in [0, 1]
    init_data = jnp.asarray(next(iter(train_ds_bpd))['image']._numpy())
    rng, step_rng = jax.random.split(rng)
    deq_model, deq_init_params = mutils.data_dependent_init_of_dequantizer(step_rng, config, init_data)
    deq_optimizer = losses.get_optimizer(config).create(deq_init_params)
    deq_state = mutils.DeqState(step=0, optimizer=deq_optimizer,
                                lr=config.optim.lr, ema_rate=config.deq.ema_rate,
                                params_ema=deq_init_params, ema_train_bpd=0,
                                ema_eval_bpd=0, rng=rng)
    deq_state = checkpoints.restore_checkpoint(os.path.join(workdir, deq_folder, "checkpoints"),
                                                deq_state, step=6)
    # deq_state = checkpoints.restore_checkpoint(os.path.join(workdir, deq_folder, "checkpoints"),
    #                                            deq_state, step=4)
    logging.info("Successfully loaded the variational dequantizer!")
    dequantizer = mutils.get_dequantizer(deq_model, deq_state.params_ema, train=False)
    p_dequantizer = jax.pmap(dequantizer, axis_name='batch')
  else:
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        additional_dim=None,
                                                        uniform_dequantization=True, evaluation=True)
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = config.eval.num_repeats
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.bound:
    likelihood_fn = bound_likelihood.get_likelihood_bound_fn(sde, score_model, inverse_scaler,
                                                              dsm=config.eval.dsm,
                                                              eps=config.training.smallest_time,
                                                              importance_weighting=True,
                                                              N=1000,
                                                              eps_offset=config.eval.offset)
  else:
    likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler, eps=config.training.smallest_time)

  @jax.pmap
  def drift_fn(state, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  num_bpd_rounds = len(ds_bpd) * bpd_num_repeats
  # Restore evaluation after pre-emption
  eval_meta = restore_eval_meta(eval_dir, num_bpd_rounds, config.eval.begin_ckpt, rng)
  rng = eval_meta.rng
  begin_ckpt = eval_meta.ckpt_id
  begin_bpd_round = eval_meta.bpd_round_id
  # Repeat multiple times to reduce variance when needed
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
    logging.info("Checkpoint filename: %s" % (ckpt_filename,))
    try:
      state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
      state = convert_checkpoint(state)
    except:
      raise RuntimeError("Unable to load checkpoint")
    # Replicate the training state for executing on multiple devices
    pstate = flax.jax_utils.replicate(state)

    bpds = []
    begin_repeat_id = begin_bpd_round // len(ds_bpd)
    begin_batch_id = begin_bpd_round % len(ds_bpd)
    rng = eval_meta.rng
    for repeat in range(begin_repeat_id, bpd_num_repeats):
      bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
      for _ in range(begin_batch_id):
        next(bpd_iter)
      for batch_id in range(begin_batch_id, len(ds_bpd)):
        bpd_round_id = batch_id + len(ds_bpd) * repeat
        bpd_npz_path = os.path.join(eval_dir,
                                    f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz")
        if tf.io.gfile.exists(bpd_npz_path):
          continue
        batch = next(bpd_iter)
        batch['image'] = batch['image'].numpy()
        batch['label'] = batch['label'].numpy()
        if block_shuffle > 0:
          logging.info("Block shuffling ...")
          rng, *perm_rng = jax.random.split(rng, batch['image'].shape[1] + 1)
          batch['image'] = randomize_tiles_shuffle_blocks_batch(batch['image'][0], block_shuffle, block_shuffle, jnp.array(perm_rng))
          batch['image'] = batch['image'][None, ...]
        eval_batch = jtu.tree_map(lambda x: scaler(x), batch)
        if config.eval.dequantizer:
          rng, step_rng = jax.random.split(rng)
          data = eval_batch['image']
          u = jax.random.normal(step_rng, data.shape)
          noise, logpd = p_dequantizer(u, inverse_scaler(data))
          data = scaler((inverse_scaler(data) * 255. + noise) / 256.)
          bpd_d = -logpd / np.log(2.)
          dim = np.prod(noise.shape[2:])
          bpd_d = bpd_d / dim
        else:
          data = eval_batch['image']
        


        rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        # bpd = likelihood_fn(step_rng, pstate, data)[0]
        drift = drift_fn(pstate, data,  sampling_eps*jnp.ones((data.shape[0]*data.shape[1],)))
        drift = drift.reshape((data.shape[0]*data.shape[1], -1))
        bpd = jnp.linalg.norm(drift, axis=1)

        # if config.eval.dequantizer:
        #   bpd = bpd + bpd_d
        # bpd = bpd.reshape(-1)
        bpds.extend(bpd)
        logging.info(
          "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, jnp.mean(jnp.asarray(bpds))))
        
        # Save bits/dim to disk or Google Cloud Storage
        with tf.io.gfile.GFile(bpd_npz_path, "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, bpd)
          fout.write(io_buffer.getvalue())

        # Save intermediate states to resume evaluation after pre-emption
        save_eval_meta(eval_dir, num_bpd_rounds, eval_meta, ckpt, bpd_round_id, rng)
      begin_batch_id = 0
    begin_bpd_round = 0

  # Remove all meta files after finishing evaluation
  # meta_files = tf.io.gfile.glob(
  #   os.path.join(eval_dir, f"meta_{jax.host_id()}_*"))
  # for file in meta_files:
  #   tf.io.gfile.remove(file)

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval", "train_deq"], "Running mode: train or eval")
flags.DEFINE_integer("block_shuffle", 0, "Block shuffle image")
flags.DEFINE_string("eval_folder", "eval_test_bpd",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("deq_folder", "flowpp_dequantizer", "The folder name for dequantizer training.")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  tf.config.experimental.set_visible_devices([], "GPU")
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  os.environ['XLA_FLAGS']='--xla_gpu_strict_conv_algorithm_picker=false'
  evaluate_bpd(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, FLAGS.deq_folder)

if __name__ == "__main__":
  app.run(main)