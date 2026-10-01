"""Microduck *bow* task — commanded stand ↔ ~45° bow (SitStand recipe).

One policy, both directions, driven by a posture flag in the twist slot:
    cmd = [bow_flag, 0, 0]   bow_flag ∈ {0 = STAND, 1 = BOW}
"Stand" is the all-zero command — same deployment idle as every other policy.
Press Y in ``infer_policy.py --bow ...`` to flip the flag (same path as
``--sitstand``).

Design:
  - SitStandCommand slews an internal blend over POSTURE_RAMP_S (anti-jackpot).
  - posture_* rewards select HOME+STAND_Z vs BOW keyframe+BOW_Z from the blend.
  - Orientation: upright only when standing; pitch tracking toward sin(45°)
    when bowing (upright would fight the bow).
  - Obs 61D: twist carries the flag; head/body slots zero-padded.

Keyframe (BAM settle 2026-10-01): hipΔ=0.65, ankle_scale=1.1, kneeΔ=0.40 →
tilt ≈ 43°, lean ≈ 0.68, trunk z ≈ 0.096. Orientation target is exact sin(45°).

NOTE: older *episodic-only* bow checkpoints (pre-commanded rewrite) are NOT
compatible — retrain after this change.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to velocity / sitstand) ────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode long enough for ≥1 full stand→bow→stand cycle (dwell 3.5–6.5 s).
EPISODE_LENGTH_S = 12.0
POSTURE_DWELL_S  = (3.5, 6.5)
POSTURE_RAMP_S   = 1.5  # slewed target traverse stand↔bow
BOW_PROB         = 0.5

# BAM-settled bow keyframe (servo joint indices).
BOWING_TARGET_OVERRIDES = {
    2:  -1.1079,  # left  hip_pitch
    3:   0.3951,  # left  knee
    4:   1.1680,  # left  ankle
    5:   0.6491,  # neck_pitch
    6:   0.6491,  # head_pitch
    11:  1.1079,  # right hip_pitch
    12: -0.3951,  # right knee
    13: -1.1680,  # right ankle
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

STAND_Z = 0.115
BOW_Z = 0.096
TARGET_LEAN = math.sin(math.radians(45))  # ≈ 0.707

MAX_DESCENT_SPEED = 0.08  # m/s downward during the bow
MAX_RISE_SPEED = 0.10     # m/s upward when returning to stand

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_bow_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Commanded stand ↔ 45° bow environment (flat terrain)."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
        "upright",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Posture-conditioned task stack (sitstand recipe, bow keyframe) ───────
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=2.0,
        params={
            "command_name": "twist",
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": BOWING_TARGET_OVERRIDES,
        },
    )
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1,
        weight=1.25,
        params={
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": BOWING_TARGET_OVERRIDES,
        },
    )

    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z": BOW_Z,
            "stand_z": STAND_Z,
            "std": 0.04,
        },
    )
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=4.0,
        params={
            "command_name": "twist",
            "sit_z": BOW_Z,
            "stand_z": STAND_Z,
        },
    )

    cfg.rewards["posture_pitch"] = RewardTermCfg(
        func=microduck_mdp.posture_pitch_gaussian,
        weight=3.0,
        params={
            "command_name": "twist",
            "bow_pitch": TARGET_LEAN,
            "stand_pitch": 0.0,
            "std": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["posture_pitch_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_pitch_gaussian,
        weight=2.0,
        params={
            "command_name": "twist",
            "bow_pitch": TARGET_LEAN,
            "stand_pitch": 0.0,
            "std": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["posture_pitch_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pitch_l1,
        weight=2.0,
        params={
            "command_name": "twist",
            "bow_pitch": TARGET_LEAN,
            "stand_pitch": 0.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Upright only while standing — must not fight the bow lean.
    cfg.rewards["upright_when_standing"] = RewardTermCfg(
        func=microduck_mdp.posture_upright_when_standing,
        weight=2.0,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    cfg.rewards["bow_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_bow_composite,
        weight=3.0,
        params={
            "command_name": "twist",
            "sit_overrides": BOWING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z": BOW_Z,
            "stand_z": STAND_Z,
            "bow_pitch": TARGET_LEAN,
            "stand_pitch": 0.0,
            "height_std": 0.04,
            "pitch_std": 0.20,
            "pose_std": 0.40,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "twist",
            "max_height": 0.125,
            "max_vz": MAX_RISE_SPEED,
        },
    )

    # Self-negating speed/impact penalties → POSITIVE weights.
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=8.0,
        params={
            "max_down_vel": MAX_DESCENT_SPEED,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["rise_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_upward_velocity_penalty,
        weight=0.0,  # introduced after the rise is discovered
        params={
            "max_up_vel": MAX_RISE_SPEED,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=1.0,
        params={"sensor_name": feet_ground_cfg.name},
    )

    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations ──────────────────────────────────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: bow/stand flag in twist (SitStandCommand) ────────────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(
        **{
            **vars(command),
            "sit_prob": BOW_PROB,
            "ramp_s": POSTURE_RAMP_S,
            "sit_z": BOW_Z,
            "stand_z": STAND_Z,
        }
    )
    cfg.commands.pop("head_pose", None)
    cfg.commands.pop("body_pose", None)

    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Standing-only resets: bowed spawn needs a pitched freejoint (joints alone
    # leave the root upright and tip). Mid-episode flag flips train both ways.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "sitting_tilt_max": math.radians(5),
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 750 * 24, "weight": -0.4},
                {"step": 1000 * 24, "weight": -0.6},
                {"step": 1250 * 24, "weight": -0.8},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "rise_speed",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 750 * 24, "weight": 5.0},
                {"step": 1500 * 24, "weight": 8.0},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {"step": 750 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg


MicroduckBowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_bow",
    run_name="microduck_bow",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=5_000,
)
