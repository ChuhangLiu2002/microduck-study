"""Cfg invariants for the episodic bow task."""

import math

from mjlab_microduck.tasks.microduck_bow_env_cfg import (
    BOW_Z,
    BOWING_TARGET_OVERRIDES,
    EPISODE_LENGTH_S,
    TARGET_LEAN,
    _LEG_JOINTS,
    make_microduck_bow_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def test_env_builds_train_and_play():
    assert make_microduck_bow_env_cfg() is not None
    assert make_microduck_bow_env_cfg(play=True) is not None


def test_episode_length():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 5.0


def test_target_is_45_deg_forward_lean():
    assert abs(TARGET_LEAN - math.sin(math.radians(45))) < 1e-9
    assert 0.09 <= BOW_Z <= 0.11


def test_bow_keyframe_is_symmetric():
    # Mirrored hip/knee/ankle signs; neck shared.
    assert BOWING_TARGET_OVERRIDES[2] == -BOWING_TARGET_OVERRIDES[11]
    assert BOWING_TARGET_OVERRIDES[3] == -BOWING_TARGET_OVERRIDES[12]
    assert BOWING_TARGET_OVERRIDES[4] == -BOWING_TARGET_OVERRIDES[13]


def test_no_upright_reward():
    cfg = make_microduck_bow_env_cfg()
    assert "upright" not in cfg.rewards
    assert "pitch_bow" in cfg.rewards
    assert "bow_composite" in cfg.rewards


def test_self_negating_penalties_use_positive_weight():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.rewards["pitch_bow_l1"].weight > 0
    assert cfg.rewards["pose_bow_l1"].weight > 0
    assert cfg.rewards["height_bow_l1"].weight > 0
    assert cfg.rewards["gentle_motion"].weight > 0


def test_walking_rewards_removed():
    cfg = make_microduck_bow_env_cfg()
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ):
        assert name not in cfg.rewards, name


def test_obs_parity_with_velocity():
    bow = make_microduck_bow_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert set(bow.observations[grp].terms) >= {
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
            "head_command",
            "body_command",
        }
    # Actor must stay 61D at runtime — head/body are zero-padded.
    assert bow.observations["actor"].terms["head_command"].params["dim"] == 4
    assert bow.observations["actor"].terms["body_command"].params["dim"] == 6
    # Velocity keeps live head/body commands; bow must not.
    assert "head_pose" not in bow.commands
    assert "body_pose" not in bow.commands
    _ = vel  # parity reference imported for future shape checks


def test_pose_rewards_point_at_bow_overrides():
    cfg = make_microduck_bow_env_cfg()
    for name in ("pose_bow_legs", "pose_bow_l1", "bow_composite"):
        assert cfg.rewards[name].params["target_overrides"] is BOWING_TARGET_OVERRIDES
    assert cfg.rewards["pose_bow_legs"].params["joint_indices"] == _LEG_JOINTS


def test_standing_reset_only():
    cfg = make_microduck_bow_env_cfg()
    p = cfg.events["set_ground_state"].params
    assert p["standing_prob"] == 1.0
    assert p["sitting_prob"] == 0.0
    assert p["face_down_prob"] == 0.0
    assert p["face_up_prob"] == 0.0


def test_expand_bam_friction_fields_present():
    cfg = make_microduck_bow_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
