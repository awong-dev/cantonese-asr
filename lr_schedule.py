"""
Shared LR schedule utilities for ASR training scripts.

Provides:
- get_tri_stage_schedule: fairseq-style warmup/hold/decay LR schedule
- TriStageCheckpointCallback: saves a checkpoint before the decay phase begins
- add_lr_schedule_args: argparse helper for LR schedule options
- resolve_lr_schedule_args: resolve args into HF scheduler config + tri_stage params
"""

import os

from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback


def get_tri_stage_schedule(
    optimizer,
    num_training_steps,
    warmup_pct=0.1,
    hold_pct=0.4,
    final_lr_scale=0.05,
):
    """
    Create a tri-stage LR schedule: linear warmup -> constant hold -> linear decay.

    This is the schedule used in the original fairseq wav2vec2 fine-tuning recipe.
    The LR ramps linearly from 0 to peak during warmup, holds at peak, then
    decays linearly to final_lr_scale * peak_lr.

    Args:
        optimizer: The optimizer to schedule
        num_training_steps: Total number of training steps
        warmup_pct: Fraction of steps for warmup (default: 0.1)
        hold_pct: Fraction of steps for hold (default: 0.4)
        final_lr_scale: Final LR as fraction of peak (default: 0.05)

    Returns:
        LambdaLR scheduler
    """
    warmup_steps = int(num_training_steps * warmup_pct)
    hold_steps = int(num_training_steps * hold_pct)
    decay_steps = num_training_steps - warmup_steps - hold_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: 0 -> 1
            return current_step / max(1, warmup_steps)
        elif current_step < warmup_steps + hold_steps:
            # Hold at peak
            return 1.0
        else:
            # Linear decay: 1 -> final_lr_scale
            decay_progress = (current_step - warmup_steps - hold_steps) / max(1, decay_steps)
            return max(final_lr_scale, 1.0 - (1.0 - final_lr_scale) * decay_progress)

    return LambdaLR(optimizer, lr_lambda)


class TriStageCheckpointCallback(TrainerCallback):
    """Save a checkpoint at the end of the hold phase, before decay begins.

    This preserves the model at peak LR before the decay phase reduces it.
    Saved to {output_dir}/pre-decay/ which is outside the Trainer's normal
    checkpoint rotation, so it won't be deleted by save_total_limit.

    After constructing the Trainer, set callback.trainer = trainer so the
    callback can call save_model() directly.
    """

    def __init__(self, num_training_steps, warmup_pct, hold_pct):
        self.decay_start_step = int(
            num_training_steps * (warmup_pct + hold_pct)
        )
        self.saved = False
        self.trainer = None  # set by caller after Trainer construction

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Tri-stage: will save pre-decay checkpoint at step {self.decay_start_step}")

    def on_step_end(self, args, state, control, **kwargs):
        if not self.saved and state.global_step >= self.decay_start_step:
            self.saved = True
            save_dir = os.path.join(args.output_dir, "pre-decay")
            print(f"\n{'=' * 60}")
            print(f"Tri-stage: saving pre-decay checkpoint at step {state.global_step}")
            print(f"  -> {save_dir}")
            print(f"{'=' * 60}")
            if self.trainer is not None:
                self.trainer.save_model(save_dir)


def add_lr_schedule_args(parser):
    """Add --lr_schedule and tri-stage-specific args to an argparse parser."""
    parser.add_argument(
        "--lr_schedule", type=str, default="cosine",
        choices=["cosine", "tri_stage"],
        help="LR schedule: 'cosine' (HF default) or 'tri_stage' (fairseq-style "
             "warmup/hold/decay). (default: cosine)",
    )
    parser.add_argument(
        "--tri_stage_warmup_pct", type=float, default=0.1,
        help="Fraction of total steps for warmup phase (default: 0.1)",
    )
    parser.add_argument(
        "--tri_stage_hold_pct", type=float, default=0.4,
        help="Fraction of total steps for hold phase (default: 0.4)",
    )
    parser.add_argument(
        "--tri_stage_final_lr_scale", type=float, default=0.05,
        help="Final LR as a fraction of peak LR (default: 0.05)",
    )


def resolve_lr_schedule_args(args, total_steps):
    """
    Given parsed args and total training steps, return (hf_scheduler_type,
    hf_warmup_steps, tri_stage_args_or_none).

    tri_stage_args is a dict suitable for passing to get_tri_stage_schedule()
    as **kwargs (alongside optimizer), or None if using cosine.
    """
    if args.lr_schedule == "tri_stage":
        hf_scheduler_type = "constant"
        hf_warmup_steps = 0
        tri_stage_args = {
            "num_training_steps": total_steps,
            "warmup_pct": args.tri_stage_warmup_pct,
            "hold_pct": args.tri_stage_hold_pct,
            "final_lr_scale": args.tri_stage_final_lr_scale,
        }
        print(f"LR schedule: tri_stage (warmup={args.tri_stage_warmup_pct:.0%}, "
              f"hold={args.tri_stage_hold_pct:.0%}, "
              f"decay={1-args.tri_stage_warmup_pct-args.tri_stage_hold_pct:.0%}, "
              f"final_lr_scale={args.tri_stage_final_lr_scale})")
    else:
        hf_scheduler_type = "cosine"
        hf_warmup_steps = args.warmup
        tri_stage_args = None
        print(f"LR schedule: cosine (warmup={args.warmup} steps)")

    return hf_scheduler_type, hf_warmup_steps, tri_stage_args
