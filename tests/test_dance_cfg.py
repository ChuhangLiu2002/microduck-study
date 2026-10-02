"""Cfg invariants for the 8 s dance task."""

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_dance_env_cfg import (
    DANCE_KNOT_PHASES,
    DANCE_KNOT_POSES,
    DANCE_PERIOD,
    EPISODE_LENGTH_S,
    STAND_POSE,
    make_microduck_dance_env_cfg,
)


def test_env_builds_train_and_play():
    assert make_microduck_dance_env_cfg() is not None
    assert make_microduck_dance_env_cfg(play=True) is not None


def test_period_and_episode_length():
    cfg = make_microduck_dance_env_cfg()
    assert DANCE_PERIOD == 8.0
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 12.0
    assert cfg.commands["twist"].period == 8.0
    assert cfg.commands["twist"].randomize_phase is True


def test_uses_phase_command():
    cfg = make_microduck_dance_env_cfg()
    assert isinstance(cfg.commands["twist"], microduck_mdp.GroundPickPhaseCommandCfg)


def test_reward_signs():
    cfg = make_microduck_dance_env_cfg()
    assert cfg.rewards["dance_pose"].weight > 0
    assert cfg.rewards["dance_pose_l1"].weight > 0  # self-negating L1
    assert cfg.rewards["dance_vel"].weight > 0
    assert cfg.rewards["feet_grounded"].weight > 0
    assert cfg.rewards["action_rate_l2"].weight < 0
    assert cfg.rewards["body_ang_vel"].weight < 0
    assert cfg.rewards["angular_momentum"].weight < 0
    assert cfg.rewards["self_collisions"].weight < 0


def test_symmetry_off():
    from mjlab_microduck.tasks.microduck_dance_env_cfg import MicroduckDanceRlCfg

    assert MicroduckDanceRlCfg.algorithm.symmetry_cfg is None


def test_keyframes_share_joint_names():
    keys = set(STAND_POSE.keys())
    assert len(DANCE_KNOT_PHASES) == len(DANCE_KNOT_POSES)
    assert DANCE_KNOT_PHASES[0] == 0.0 and DANCE_KNOT_PHASES[-1] == 1.0
    for pose in DANCE_KNOT_POSES:
        assert set(pose.keys()) == keys


def test_dance_velocity_target_segments():
    phase = torch.tensor([0.05, 0.20, 0.40, 0.60, 0.75, 0.90])
    v = microduck_mdp.dance_velocity_target(phase)
    assert v.shape == (6, 3)
    assert torch.allclose(v[0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(v[1], torch.tensor([0.08, 0.0, 0.0]))
    assert torch.allclose(v[2], torch.tensor([0.0, 0.06, 0.0]))
    assert torch.allclose(v[3], torch.tensor([0.0, -0.06, 0.0]))
    assert torch.allclose(v[4], torch.tensor([-0.06, 0.0, 0.3]))
    assert torch.allclose(v[5], torch.tensor([0.0, 0.0, 0.0]))


def test_dance_pose_target_endpoints():
    stand = torch.tensor([float(STAND_POSE[k]) for k in sorted(STAND_POSE)])
    other = stand + 0.1
    phase = torch.tensor([0.0, 1.0])
    out = microduck_mdp.dance_pose_target(phase, [0.0, 1.0], [stand, other])
    assert torch.allclose(out[0], stand)
    assert torch.allclose(out[1], other)


def test_head_body_command_padded():
    cfg = make_microduck_dance_env_cfg()
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4
        assert cfg.observations[group].terms["body_command"].params["dim"] == 6
