"""Cfg invariants for the commanded stand↔bow task."""

import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import (
    BOW_Z,
    BOWING_TARGET_OVERRIDES,
    EPISODE_LENGTH_S,
    STAND_Z,
    TARGET_LEAN,
    _LEG_JOINTS,
    make_microduck_bow_env_cfg,
)


def test_env_builds_train_and_play():
    assert make_microduck_bow_env_cfg() is not None
    assert make_microduck_bow_env_cfg(play=True) is not None


def test_episode_length():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 12.0


def test_target_is_45_deg_forward_lean():
    assert abs(TARGET_LEAN - math.sin(math.radians(45))) < 1e-9
    assert 0.09 <= BOW_Z <= 0.11
    assert STAND_Z == 0.115


def test_bow_keyframe_is_symmetric():
    assert BOWING_TARGET_OVERRIDES[2] == -BOWING_TARGET_OVERRIDES[11]
    assert BOWING_TARGET_OVERRIDES[3] == -BOWING_TARGET_OVERRIDES[12]
    assert BOWING_TARGET_OVERRIDES[4] == -BOWING_TARGET_OVERRIDES[13]


def test_uses_sitstand_command():
    cfg = make_microduck_bow_env_cfg()
    assert isinstance(cfg.commands["twist"], microduck_mdp.SitStandCommandCfg)
    assert cfg.commands["twist"].sit_z == BOW_Z
    assert cfg.commands["twist"].stand_z == STAND_Z


def test_no_bare_upright_reward():
    cfg = make_microduck_bow_env_cfg()
    assert "upright" not in cfg.rewards
    assert "upright_when_standing" in cfg.rewards
    assert "posture_pitch" in cfg.rewards
    assert "bow_composite" in cfg.rewards


def test_self_negating_penalties_use_positive_weight():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.rewards["posture_pitch_l1"].weight > 0
    assert cfg.rewards["posture_pose_l1"].weight > 0
    assert cfg.rewards["posture_height_l1"].weight > 0
    assert cfg.rewards["gentle_motion"].weight > 0
    assert cfg.rewards["descent_speed"].weight > 0


def test_pose_rewards_use_bow_overrides():
    cfg = make_microduck_bow_env_cfg()
    for name in ("posture_pose_legs", "posture_pose_l1", "bow_composite"):
        assert cfg.rewards[name].params["sit_overrides"] is BOWING_TARGET_OVERRIDES
    assert cfg.rewards["posture_pose_legs"].params["joint_indices"] == _LEG_JOINTS


def test_obs_command_slots_padded():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.observations["actor"].terms["head_command"].params["dim"] == 4
    assert cfg.observations["actor"].terms["body_command"].params["dim"] == 6
    assert "head_pose" not in cfg.commands
    assert "body_pose" not in cfg.commands


def test_standing_reset_only():
    cfg = make_microduck_bow_env_cfg()
    p = cfg.events["set_ground_state"].params
    assert p["standing_prob"] == 1.0
    assert p["sitting_prob"] == 0.0


def test_expand_bam_friction_fields_present():
    cfg = make_microduck_bow_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
