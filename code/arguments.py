import argparse
import os
import deepspeed
import numpy as np
from distiller import Distiller


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-path', type=str, help='model path')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--model-type", type=str, default="gpt2")
    group.add_argument("--teacher-model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--teacher-model-path", type=str)
    group.add_argument("--teacher-model-fp16", action="store_true")
    group.add_argument("--model-parallel", action="store_true")
    group.add_argument("--model-parallel-size", type=int, default=None)
    group.add_argument("--no-value", action="store_true")
    group.add_argument("--dropout-path-rate", type=float, default=None)
    group.add_argument("--fp32", action="store_true")
    group.add_argument("--model-dtype", type=str, default="fp16")

    # Add
    group.add_argument("--hidden-dim-student", type=int, default=768)
    group.add_argument("--hidden-dim-teacher", type=int, default=2048)
    group.add_argument("--top_k_vocab", type=int, default=300)
    group.add_argument("--proj_dim", type=int, default=512)
    group.add_argument("--max-student-len", type=int, default=512)
    group.add_argument("--max-teacher-len", type=int, default=1024)
    group.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="Training precision: fp32, fp16, or bf16")
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--task", type=str, default=None)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--start-epoch', type=int, default=1,
                       help='1-based epoch index to start training (resume).')
    group.add_argument('--resume-global-step', type=int, default=0,
                       help='Completed optimizer steps; cosine LR is fast-forwarded to match.')
    group.add_argument('--save-dir', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument('--save-interval', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--save-additional-suffix", type=str, default="")
    group.add_argument("--save-rollout", action="store_true")
    group.add_argument("--eb-sample-times", type=int, default=3)
    group.add_argument("--keep-best-n-checkpoints", type=int, default=3)
    group.add_argument("--criterion", type=str, default="cross_entropy")
    group.add_argument("--eval-tqdm", action="store_true")
    group.add_argument("--report-logits", action="store_true")
    group.add_argument("--only-save-projector", action="store_true")
    group.add_argument("--debug", action="store_true")
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--gen-num", type=int, default=-1)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--min-prompt-length", type=int, default=128)
    group.add_argument("--json-data", action="store_true")
    group.add_argument("--bin-data", action="store_true")
    group.add_argument("--txt-data", action="store_true")
    
    group.add_argument("--prompt-data-dir", type=str)
    group.add_argument("--pretrain-data-dir", type=str)
    group.add_argument("--eval-ppl", action="store_true")
    group.add_argument("--eval-rw", action="store_true")
    group.add_argument("--eval-gen", action="store_true")
    
    group.add_argument("--only-prompt", action="store_true")
    return parser


def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--total-iters', type=int, default=None,
                       help='total number of iterations')
    group.add_argument('--train-iters-per-epoch', type=int, default=-1,
                       help='total number of iterations per epoch')
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--seed-ppo", type=int, default=42)
    group.add_argument("--seed-lm", type=int, default=7)
    group.add_argument('--num-epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    group.add_argument('--training-epochs', type=int, default=10000)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--attn-dtype", default=None)
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument("--lr-min", type=float, default=0.0000001)
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')
    group.add_argument("--kd-rate", type=float, default=2.5)
    group.add_argument("--kd-temperature", type=float, default=1.0)
    group.add_argument("--kd-objective", type=str, default="forward_kl")
    group.add_argument("--teacher-temperature", type=float, default=1.0)
    group.add_argument("--label-smoothing", type=float, default=0.0)
    group.add_argument("--adaptive-kl-alpha", type=float, default=0.5)
    group.add_argument("--skew-lambda", type=float, default=0.1)

    # ResidualKD hyperparameters
    group.add_argument("--residualkd-lambda-res", type=float, default=0.5,
                       help="Weight for residual loss in ResidualKD")
    group.add_argument("--residualkd-lambda-warmup", type=int, default=50,
                       help="Number of steps to linearly warm up residual lambda")
    group.add_argument("--residualkd-d-bottleneck", type=int, default=64,
                       help="Anchor bottleneck dimension for ResidualKD projectors")
    group.add_argument("--residualkd-projector-load-path", type=str, default=None,
                       help="Path to pretrained ProjectorTA from Stage 1")
    group.add_argument("--residualkd-cross-tokenizer", action="store_true",
                       help="Use entropy-based residual mask for cross-tokenizer settings")

    # SRA hyperparameters
    group.add_argument("--sra-alpha", type=float, default=0.5,
                       help="Weight for hard label loss in SRA (1-alpha for KD losses)")
    group.add_argument("--sra-geom-weight", type=float, default=50.0,
                       help="Weight for geometric (relation-based) loss")
    group.add_argument("--sra-span-power", type=float, default=1.0,
                       help="Power p for span/attention weight normalization")
    group.add_argument("--sra-span-loss", action="store_true",
                       help="Enable span hidden state alignment loss")
    group.add_argument("--sra-student-layers", type=str, default=None,
                       help="Comma-separated student layer indices for span extraction (e.g. '4,8,12')")
    group.add_argument("--sra-teacher-layers", type=str, default=None,
                       help="Comma-separated teacher layer indices for span extraction (e.g. '8,16,24')")
    group.add_argument("--sra-hidden-loss-weights", type=str, default=None,
                       help="Comma-separated per-layer loss weights (e.g. '0.3,0.3,0.4')")

    # ALM hyperparameters
    group.add_argument("--alm-binarization-temp", type=float, default=100.0,
                       help="Temperature for ALM binary CE sharpening")
    group.add_argument("--alm-bias-threshold", type=float, default=0.1,
                       help="Space probability threshold for chunk merging in ALM")
    group.add_argument("--alm-loss-weight", type=float, default=3.0,
                       help="Weight multiplier for ALM distillation loss")
    group.add_argument("--alm-mode", type=str, default="merge_by_space_prob+append_space",
                       help="ALM loss mode (tokenkit: merge_by_space_prob+append_space)")
    group.add_argument("--alm-alignment", type=str, default="unconstrained",
                       choices=["unconstrained", "unbiased"],
                       help="Byte alignment for ALM: unconstrained (tokenkit default) or "
                            "unbiased (needs --tokenizer-pair-data-path)")
    group.add_argument("--tokenizer-pair-data-path", type=str, default=None,
                       help="Dir with bias1_matrix.npz, bias2_matrix.npz for unbiased alignment")
    group.add_argument("--multitask-aggregation-fn", type=str, default="approx_gradmag_preserve_mag",
                       choices=["none", "approx_gradmag", "approx_gradmag_preserve_mag"],
                       help="GradMag loss aggregation (tokenkit). 'none' = manual alm_loss_weight sum")

    # IMPACT hyperparameters
    group.add_argument("--impact-lambda", type=float, default=1.0,
                       help="Weight for IMPACT alignment loss")
    group.add_argument("--impact-top-k", type=int, default=4,
                       help="Number of top teacher layers to select by BI score")
    group.add_argument("--impact-bi-tau", type=float, default=1.0,
                       help="Temperature for BI score softmax weighting")
    group.add_argument("--impact-lambda-reg", type=float, default=1.0,
                       help="Regularization strength for vocabulary projection")
    group.add_argument(
        "--impact-choose-layer",
        "--choose-layer",
        dest="impact_choose_layer",
        type=str,
        default="BI",
        choices=["BI", "random", "first", "last", "PPL"],
        help="Teacher layer selection when impact choose_align=BI",
    )
    group.add_argument(
        "--impact-choose-align",
        "--choose-align",
        dest="impact_choose_align",
        type=str,
        default="BI",
        choices=["BI", "1-1", "1-all"],
        help="IMPACT layer alignment: BI, 1-1 (depth-ratio), 1-all (ALP-KD attention, GLMKD)",
    )

    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--num-epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument("--scheduler-name", type=str, default="constant_trm")

    return parser


def add_gen_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)
    group.add_argument("--eval-gen-repeat-times", type=int, default=3)
    
    return parser


def add_peft_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--peft", type=str, default=None)
    group.add_argument("--peft-lora-r", type=int, default=16)
    group.add_argument("--peft-lora-alpha", type=int, default=64)
    group.add_argument("--peft-lora-dropout", type=float, default=0.1)
    group.add_argument("--peft-name", type=str, default=None)
    group.add_argument("--peft-path", type=str, default=None)
    group.add_argument("--teacher-peft-name", type=str, default=None)
    group.add_argument("--teacher-peft-path", type=str, default=None)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_gen_args(parser)
    parser = add_peft_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    parser = Distiller.add_distiller_args(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
    args.n_gpu = args.n_gpu * args.n_nodes
        
    return args
