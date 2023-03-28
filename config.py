import argparse
from pathlib import Path

import torch.nn.functional as F

parser = argparse.ArgumentParser(description="PlaNet or Dreamer")
parser.add_argument("--algo", type=str, help="planet or dreamer")
parser.add_argument("--id", type=str, help="Experiment ID")
parser.add_argument("--seed", type=int, metavar="S", help="Random seed")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--env",
    type=str,
    choices=[],
    help="Gym/Control Suite environment",
)
parser.add_argument("--symbolic-env", action="store_true", help="Symbolic features")
parser.add_argument(
    "--max-episode-length",
    type=int,
    metavar="T",
    help="Max episode length",
)
parser.add_argument(
    "--experience-size",
    type=int,
    metavar="D",
    help="Experience replay size",
)  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument(
    "--cnn-activation-function",
    type=str,
    choices=dir(F),
    help="Model activation function for a convolution layer",
)
parser.add_argument(
    "--dense-activation-function",
    type=str,
    choices=dir(F),
    help="Model activation function a dense layer",
)
parser.add_argument(
    "--embedding-size",
    type=int,
    metavar="E",
    help="Observation embedding size",
)  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument(
    "--hidden-size", type=int, metavar="H", help="Hidden size"
)
parser.add_argument(
    "--belief-size", type=int, metavar="H", help="Belief/hidden size"
)
parser.add_argument(
    "--state-size", type=int, metavar="Z", help="State/latent size"
)
parser.add_argument(
    "--action-repeat", type=int, metavar="R", help="Action repeat"
)
parser.add_argument(
    "--action-noise", type=float, metavar="ε", help="Action noise"
)
parser.add_argument(
    "--episodes", type=int, metavar="E", help="Total number of episodes"
)
parser.add_argument(
    "--seed-episodes", type=int, metavar="S", help="Seed episodes"
)
parser.add_argument(
    "--collect-interval", type=int, metavar="C", help="Collect interval"
)
parser.add_argument(
    "--batch-size", type=int, metavar="B", help="Batch size"
)
parser.add_argument(
    "--chunk-size", type=int, metavar="L", help="Chunk size"
)
parser.add_argument(
    "--worldmodel-LogProbLoss",
    action="store_true",
    help="use LogProb loss for observation_model and reward_model training",
)
parser.add_argument(
    "--overshooting-distance",
    type=int,
    metavar="D",
    help="Latent overshooting distance/latent overshooting weight for t = 1",
)
parser.add_argument(
    "--overshooting-kl-beta",
    type=float,
    metavar="β>1",
    help="Latent overshooting KL weight for t > 1 (0 to disable)",
)
parser.add_argument(
    "--overshooting-reward-scale",
    type=float,
    metavar="R>1",
    help="Latent overshooting reward prediction weight for t > 1 (0 to disable)",
)
parser.add_argument(
    "--global-kl-beta",
    type=float,
    metavar="βg",
    help="Global KL weight (0 to disable)",
)
parser.add_argument("--free-nats", type=float, metavar="F", help="Free nats")
parser.add_argument(
    "--bit-depth",
    type=int,
    metavar="B",
    help="Image bit depth (quantisation)",
)
parser.add_argument(
    "--model_learning-rate", type=float,  metavar="α", help="Learning rate"
)
parser.add_argument(
    "--actor_learning-rate", type=float,  metavar="α", help="Learning rate"
)
parser.add_argument(
    "--value_learning-rate", type=float,  metavar="α", help="Learning rate"
)
parser.add_argument(
    "--learning-rate-schedule",
    type=int,
    metavar="αS",
    help="Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)",
)
parser.add_argument(
    "--adam-epsilon",
    type=float,
    
    metavar="ε",
    help="Adam optimizer epsilon value",
)
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument(
    "--grad-clip-norm",
    type=float,
    
    metavar="C",
    help="Gradient clipping norm",
)
parser.add_argument(
    "--planning-horizon",
    type=int,
    metavar="H",
    help="Planning horizon distance",
)
parser.add_argument(
    "--discount",
    type=float,
    
    metavar="H",
    help="Planning horizon distance",
)
parser.add_argument(
    "--disclam",
    type=float,
    
    metavar="H",
    help="discount rate to compute return",
)
parser.add_argument(
    "--optimisation-iters",
    type=int,
    metavar="I",
    help="Planning optimisation iterations",
)
parser.add_argument(
    "--candidates",
    type=int,
    metavar="J",
    help="Candidate samples per iteration",
)
parser.add_argument(
    "--top-candidates",
    type=int,
    metavar="K",
    help="Number of top candidates to fit",
)
parser.add_argument("--test", action="store_true", help="Test only")
parser.add_argument(
    "--test-interval",
    type=int,
    metavar="I",
    help="Test interval (episodes)",
)
parser.add_argument(
    "--test-episodes", type=int, metavar="E", help="Number of test episodes"
)
parser.add_argument(
    "--checkpoint-interval",
    type=int,
    metavar="I",
    help="Checkpoint interval (episodes)",
)
parser.add_argument(
    "--checkpoint-experience", action="store_true", help="Checkpoint experience replay"
)
parser.add_argument(
    "--models", type=str, metavar="M", help="Load model checkpoint"
)
parser.add_argument(
    "--experience-replay",
    type=str,
    metavar="ER",
    help="Load experience replay",
)
parser.add_argument("--render", action="store_true", help="Render environment")
parser.add_argument(
    "--paths-to-sample",
    type=int,
    metavar="N",
    help="Number of paths to sample in BPS",
)
parser.add_argument(
    "--violation-threshold",
    type=int,
    metavar="ε",
    help="Violation threshold in BPS",
)
parser.add_argument(
    "--path", 
    type=Path,
    default=Path('configs/default.yaml'),
    help="Path to config file"
)