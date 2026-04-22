from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAB_ROOT = ROOT / "dab_ttm"
RUN_DIR = ROOT / "last_resort" / "results" / "fixed_direct_grad_dlp38"

# Frozen configuration copied from dab_ttm/results/dlp_38/conf.yaml.
CONF: dict = {
    "k_val": 250,
    "proposal_temp": 0.1,
    "initialization": "zero",
    "initialization_noise_rate": 0.5,
    "weight_val": 0.8,
    "weight_schedule": {"name": "constant", "delta": 0.0},
    "use_scale_weights": "full",
    "debug_trace_sequences": False,
    "loss_aggregation": "none",
    "bias_update_mode": "direct_grad_norm_matched",
    "direct_grad_ar_event_only": True,
    "direct_grad_norm_match_eps": 1.0e-8,
    "direct_grad_norm_match_masked_full": True,
    "direct_grad_norm_match_topk": True,
    "direct_grad_norm_match_topk_k": 0,
    "direct_grad_norm_match_ratio_min": 0.001,
    "direct_grad_norm_match_ratio_max": 10000.0,
    "selection_lm_loss_threshold": 1.0,
    "selection_attr_convergence_threshold": 0.8,
    "experiment_type": "ttm",
    "device": "cuda",
    "save_dir": "results",
    "base_model_args": {
        "model_name_or_path": "stanford-crfm/music-large-800k",
        "trust_remote_code": True,
    },
    "discriminator": {
        "type": "distilled_clamp",
        "distilled_ckpt": "/home/hhhhaura/dmir_lab/CTRL-M/ctrlm-ismir/distilled_clamp/outputs_mlp_head/checkpoints/input_perceiver_lat4/step_00076800.pt",
        "distilled_cfg": "/home/hhhhaura/dmir_lab/CTRL-M/ctrlm-ismir/distilled_clamp/configs/mlp_head.yaml",
        "distilled_root": "/home/hhhhaura/dmir_lab/CTRL-M/ctrlm-ismir/distilled_clamp",
        "clamp3_root": "/home/hhhhaura/dmir_lab/CTRL-M//clamp3",
        "clamp3_text_model": "FacebookAI/xlm-roberta-base",
        "clamp3_weights_path": "/home/hhhhaura/dmir_lab/CTRL-M//clamp-anti-bridge/weights/weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth",
        "attr_weight": 1.0,
        "attr_loss_type": "cosine",
        "lm_reg_weight": 0.4,
        "bias_reg_weight": 0.0,
    },
    "prompt_csv": "/home/hhhhaura/dmir_lab/CTRL-M/ctrlm-ismir/dab_ttm/prompts/test_subset_prompts.csv",
    "metrics": {
        "clap": True,
        "clap_ckpt": "/home/hhhhaura/dmir_lab/CTRL-M/ctrlm-ismir/dab_ttm/assets/weights/music_speech_audioset_epoch_15_esc_89.98.pt",
        "fad": True,
        "fad_ref_audio_cache_dir": "/home/hhhhaura/dmir_lab/CTRL-M/ctrlm-ismir/dab_ttm/assets/fad_ref_audio",
    },
    "start_idx": 0,
    "end_idx": -1,
    "per_sample_batch_size": 1,
    "prompt_batch_size": 1,
    "seq_len": 1023,
    "ctrl_block_length": 510,
    "max_rolling_blocks": 20,
    "num_steps_first_block": 10,
    "num_steps_cont_block": 5,
    "temperature": 1.0,
    "top_p": 0.98,
    "do_sample": True,
    "save_midi": True,
    "save_wav": True,
    "sound_font": "",
    "wav_sample_rate": 44100,
    "sampler": "dlp",
}

# Fixed direct-grad runner controls.
SEED = 42
TRACE_NUM_STEPS = int(CONF["num_steps_first_block"])
MAX_CTX = int(CONF["seq_len"])
RENDER_EVERY_STEP = True
MAX_RENDER_STEPS = 0
SAVE_WAV = bool(CONF["save_wav"])
SAVE_MIDI = bool(CONF["save_midi"])


def assert_frozen_constraints() -> None:
    if str(CONF.get("use_scale_weights")) != "full":
        raise ValueError("last_resort requires use_scale_weights='full'.")
    if str(CONF.get("bias_update_mode")) != "direct_grad_norm_matched":
        raise ValueError("last_resort requires bias_update_mode='direct_grad_norm_matched'.")
    if int(CONF.get("per_sample_batch_size", 1)) != 1:
        raise ValueError("last_resort requires per_sample_batch_size=1.")
    if int(CONF.get("prompt_batch_size", 1)) != 1:
        raise ValueError("last_resort requires prompt_batch_size=1.")
