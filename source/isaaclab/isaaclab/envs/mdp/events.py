# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression "/World/envs/env_.*/Object" has a child
    with the path "/World/envs/env_.*/Object/mesh", then the relative child path should be "mesh" or
    "/mesh".

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # sample scale values
    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

# class randomize_rigid_body_material(ManagerTermBase):
#     """Randomize the physics materials on all geometries of the asset.

#     This function creates a set of physics materials with random static friction, dynamic friction, and restitution
#     values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
#     uniform random values from the given ranges.

#     The material properties are then assigned to the geometries of the asset. The assignment is done by
#     creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
#     is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
#     all bodies). The integer values are used as indices to select the material properties from the
#     material buckets.

#     If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
#     the static friction. This obeys the physics constraint on friction values. However, it may not always be
#     essential for the application. Thus, the flag is set to ``False`` by default.

#     .. attention::
#         This function uses CPU tensors to assign the material properties. It is recommended to use this function
#         only during the initialization of the environment. Otherwise, it may lead to a significant performance
#         overhead.

#     .. note::
#         PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
#         limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
#         Afterwards, these materials are randomly assigned to the geometries of the asset.
#     """

#     def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
#         """Initialize the term.

#         Args:
#             cfg: The configuration of the event term.
#             env: The environment instance.

#         Raises:
#             ValueError: If the asset is not a RigidObject or an Articulation.
#         """
#         super().__init__(cfg, env)

#         # extract the used quantities (to enable type-hinting)
#         self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
#         self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

#         if not isinstance(self.asset, (RigidObject, Articulation)):
#             raise ValueError(
#                 f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
#                 f" with type: '{type(self.asset)}'."
#             )

#         # obtain number of shapes per body (needed for indexing the material properties correctly)
#         # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
#         #  per body. We use the physics simulation view to obtain the number of shapes per body.
#         if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
#             self.num_shapes_per_body = []
#             for link_path in self.asset.root_physx_view.link_paths[0]:
#                 link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
#                 self.num_shapes_per_body.append(link_physx_view.max_shapes)
#             # ensure the parsing is correct
#             num_shapes = sum(self.num_shapes_per_body)
#             expected_shapes = self.asset.root_physx_view.max_shapes
#             if num_shapes != expected_shapes:
#                 raise ValueError(
#                     "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
#                     f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
#                 )
#         else:
#             # in this case, we don't need to do special indexing
#             self.num_shapes_per_body = None

#         # obtain parameters for sampling friction and restitution values
#         static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
#         dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
#         restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
#         num_buckets = int(cfg.params.get("num_buckets", 1))

#         # sample material properties from the given ranges
#         # note: we only sample the materials once during initialization
#         #   afterwards these are randomly assigned to the geometries of the asset
#         range_list = [static_friction_range, dynamic_friction_range, restitution_range]
#         ranges = torch.tensor(range_list, device="cpu")
#         self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

#         # ensure dynamic friction is always less than static friction
#         make_consistent = cfg.params.get("make_consistent", False)
#         if make_consistent:
#             self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

#     def __call__(
#         self,
#         env: ManagerBasedEnv,
#         env_ids: torch.Tensor | None,
#         static_friction_range: tuple[float, float],
#         dynamic_friction_range: tuple[float, float],
#         restitution_range: tuple[float, float],
#         num_buckets: int,
#         asset_cfg: SceneEntityCfg,
#         make_consistent: bool = False,
#     ):
#         # resolve environment ids
#         if env_ids is None:
#             env_ids = torch.arange(env.scene.num_envs, device="cpu")
#         else:
#             env_ids = env_ids.cpu()

#         # randomly assign material IDs to the geometries
#         total_num_shapes = self.asset.root_physx_view.max_shapes
#         bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
#         material_samples = self.material_buckets[bucket_ids]

#         # retrieve material buffer from the physics simulation
#         materials = self.asset.root_physx_view.get_material_properties()

#         # update material buffer with new samples
#         if self.num_shapes_per_body is not None:
#             # sample material properties from the given ranges
#             for body_id in self.asset_cfg.body_ids:
#                 # obtain indices of shapes for the body
#                 start_idx = sum(self.num_shapes_per_body[:body_id])
#                 end_idx = start_idx + self.num_shapes_per_body[body_id]
#                 # assign the new materials
#                 # material samples are of shape: num_env_ids x total_num_shapes x 3
#                 materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
#         else:
#             # assign all the materials
#             materials[env_ids] = material_samples[:]

#         # apply to simulation
#         self.asset.root_physx_view.set_material_properties(materials, env_ids)

import time

class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction. This obeys the physics constraint on friction values. However, it may not always be
    essential for the application. Thus, the flag is set to ``False`` by default.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
        Afterwards, these materials are randomly assigned to the geometries of the asset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # ---- fixed global levels (bins) ----
        dyn_lo, dyn_hi = cfg.params.get("dynamic_friction_range", (0.2, 2.0))
        self.restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        self.num_buckets = int(cfg.params.get("num_buckets", 8))

        # endpoints included: [dyn_lo, ..., dyn_hi]
        if self.num_buckets < 2:
            raise ValueError("num_buckets must be >= 2 to include both endpoints.")
        self.friction_levels = torch.linspace(float(dyn_lo), float(dyn_hi), steps=self.num_buckets)

        # simple deterministic cycle so env->level changes every call
        self._cycle = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = True,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        E = len(env_ids)
        S = self.asset.root_physx_view.max_shapes
        L = self.num_buckets

        # fixed global bins (unchanged)
        levels = self.friction_levels  # [L]

        # --- PER-RESET MAPPING (stateless, time-based) ---
        # 1) make a fresh permutation of the L bins using a time-derived seed
        g = torch.Generator(device="cpu").manual_seed(int(time.time_ns() & 0xFFFFFFFF))
        perm_levels = torch.randperm(L, generator=g)

        # 2) repeat/truncate to cover all envs, then assign in the order of env_ids
        assigned_ids = perm_levels.repeat((E + L - 1) // L)[:E]   # [E]
        fric_env = levels[assigned_ids]                            # [E]

        # 3) restitution per-env (constant or random with same stateless generator)
        r_lo, r_hi = self.restitution_range
        rest_env = torch.full((E,), float(r_lo)) if r_lo == r_hi else torch.empty(E).uniform_(float(r_lo), float(r_hi), generator=g)

        # build [static==dynamic, dynamic, restitution] and broadcast to shapes
        env_samples = torch.stack([fric_env, fric_env, rest_env], dim=1)           # [E, 3]

        # Print friction
        # print('env_samples', env_samples)
        material_samples = env_samples[:, None, :].expand(E, S, 3)                 # [E, S, 3]

        # print("reset mapping: env_ids[:5] -> assigned_ids[:5] -> fric[:5]",
        #     env_ids[:5].tolist(), assigned_ids[:5].tolist(), fric_env[:5].tolist())
        # print("friction levels (global, fixed):", levels.tolist())

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)



def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,
):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    If the ``recompute_inertia`` flag is set to ``True``, the function recomputes the inertia tensor of the bodies
    after setting the mass. This is useful when the mass is changed significantly, as the inertia tensor depends
    on the mass. It assumes the body is a uniform density object. If the body is not a uniform density object,
    the inertia tensor may not be accurate.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # apply randomization on default values
    # this is to make sure when calling the function multiple times, the randomization is applied on the
    # default values and not the previously randomized values
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

    # sample from the given range
    # note: we modify the masses in-place for all environments
    #   however, the setter takes care that only the masses of the specified environments are modified
    masses = _randomize_prop_by_op(
        masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )

    # set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids)

    # recompute inertia tensors if needed
    if recompute_inertia:
        # compute the ratios of the new masses to the initial masses
        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        # scale the inertia tensors by the the ratios
        # since mass randomization is done on default values, we can use the default inertia tensors
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            # inertia has shape: (num_envs, num_bodies, 9) for articulation
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            # inertia has shape: (num_envs, 9) for rigid object
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        # set the inertia tensors into the physics simulation
        asset.root_physx_view.set_inertias(inertias, env_ids)

def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random offset."""
    # Extract the asset (assumed to be a RigidObject)
    asset = env.scene[asset_cfg.name]
    assert isinstance(asset, RigidObject), "This function is for RigidObjects only"

    # Resolve environment ids (CPU tensors only)
    env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()

    # Cache original values on first call to prevent drift
    if not hasattr(asset, "_original_coms"):
        asset._original_coms = asset.root_physx_view.get_coms().clone()
        asset._original_inertias = asset.root_physx_view.get_inertias().clone()
        asset._original_masses = asset.root_physx_view.get_masses().clone()

    # Generate random offsets [num_selected_envs, 3]
    range_tensor = torch.tensor([
        com_range.get("x", (0.0, 0.0)),
        com_range.get("y", (0.0, 0.0)), 
        com_range.get("z", (0.0, 0.0))
    ], device="cpu")
    
    rand_samples = math_utils.sample_uniform(
        range_tensor[:, 0], 
        range_tensor[:, 1], 
        (len(env_ids), 3),  # Shape matches per-env CoM dimension
        device="cpu"
    )

    # Apply offsets to original COM (can't use arbitrary absolute positions)
    coms = asset._original_coms.clone()
    # print("original coms", coms[env_ids])
    
    # Apply sampled offsets to original COM positions
    coms[env_ids, :3] += rand_samples

    # print("new coms (original + offset)", coms[env_ids])
    
    # Set new CoMs
    asset.root_physx_view.set_coms(coms, env_ids)

    # Update inertia tensors using parallel axis theorem
    # Start from original inertias and masses to prevent accumulation
    masses = asset._original_masses.clone()
    inertias = asset._original_inertias.clone()

    for idx, env_id in enumerate(env_ids):
        d = rand_samples[idx]  # Offset for this environment [3]
        m = masses[env_id]     # Mass for this environment
        
        # Parallel axis theorem: ΔI = m*(||d||²·I - ddᵀ)
        d_outer = torch.outer(d, d)
        d_norm_sq = torch.dot(d, d)
        shift_tensor = m * (d_norm_sq * torch.eye(3) - d_outer)
        
        # Update inertia tensor from original
        I_orig = asset._original_inertias[env_id].view(3, 3)
        I_new = I_orig + shift_tensor
        inertias[env_id] = I_new.reshape(-1)

    # Apply updated inertias
    asset.root_physx_view.set_inertias(inertias, env_ids)

# def randomize_rigid_body_com(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     com_range: dict[str, tuple[float, float]],
#     asset_cfg: SceneEntityCfg,
# ):
#     """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

#     .. note::
#         This function uses CPU tensors to assign the CoM. It is recommended to use this function
#         only during the initialization of the environment.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # resolve environment ids
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device="cpu")
#     else:
#         env_ids = env_ids.cpu()

#     # resolve body indices
#     if asset_cfg.body_ids == slice(None):
#         body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
#     else:
#         body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

#     # sample random CoM values
#     range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
#     ranges = torch.tensor(range_list, device="cpu")
#     rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

#     # get the current com of the bodies (num_assets, num_bodies)
#     coms = asset.root_physx_view.get_coms().clone()

#     # Randomize the com in range
#     coms[:, body_ids, :3] += rand_samples

#     # Set the new coms
#     asset.root_physx_view.set_coms(coms, env_ids)
    
#     # Print first env com
#     print("coms:", coms[0, :, :])

# def randomize_rigid_body_com(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     com_range: dict[str, tuple[float, float]],
#     asset_cfg: SceneEntityCfg,
# ):
#     """Randomize the COM of a RigidObject and update inertia about the NEW COM.
#     No mass change, no extra weight; this is a pure COM shift.
#     """
#     # Get asset
#     asset = env.scene[asset_cfg.name]
#     assert isinstance(asset, RigidObject), "This function is for RigidObjects only"

#     # Resolve env ids (CPU long)
#     dev = torch.device("cpu")
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device=dev, dtype=torch.long)
#     else:
#         env_ids = env_ids.to(device=dev, dtype=torch.long)

#     # Cache originals once (COM-frame tensors)
#     if not hasattr(asset, "_orig_massprops"):
#         asset._original_coms     = asset.root_physx_view.get_coms().clone().to(dev)      # (N,3)
#         asset._original_inertias = asset.root_physx_view.get_inertias().clone().to(dev)  # (N,9) row-major
#         asset._original_masses   = asset.root_physx_view.get_masses().clone().to(dev)    # (N,)

#     C_orig = asset._original_coms
#     J_orig = asset._original_inertias.view(-1, 3, 3)
#     m_orig = asset._original_masses

#     # Sample per-env COM offsets d ~ U(range)
#     rng = torch.tensor([
#         com_range.get("x", (0.0, 0.0)),
#         com_range.get("y", (0.0, 0.0)),
#         com_range.get("z", (0.0, 0.0)),
#     ], dtype=torch.float32, device=dev)

#     d = math_utils.sample_uniform(
#         rng[:, 0], rng[:, 1],
#         (len(env_ids), 3),
#         device=dev
#     ).to(torch.float32)  # (E,3)

#     # New COMs
#     C_new = C_orig.clone()
#     C_new[env_ids] = C_orig[env_ids] + d

#     # Update inertia: J_C' = J_C - m (||d||^2 I - d d^T)
#     eye3 = torch.eye(3, dtype=torch.float32, device=dev)
#     J_new = J_orig.clone()
#     for k, e in enumerate(env_ids.tolist()):
#         di = d[k]                                  # (3,)
#         di2 = torch.dot(di, di).item()
#         shift = m_orig[e].item() * (di2 * eye3 - torch.outer(di, di))
#         J_new[e] = J_orig[e] - shift               # MINUS for COM→COM shift

#     # Apply only to selected envs
#     asset.root_physx_view.set_coms(C_new[env_ids], env_ids)
#     asset.root_physx_view.set_inertias(J_new[env_ids].reshape(-1, 9), env_ids)

# def randomize_rigid_body_com_tblock(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     asset_cfg: SceneEntityCfg,
#     *,
#     # --- taped weight (point mass) ---
#     weight_mass_range: tuple[float, float] = (0.0, 0.0),            # kg; include 0 to sometimes have no weight
#     weight_offset_range: dict[str, tuple[float, float]] | None = None,  # meters from ORIGINAL COM; {'x':(a,b),'y':(c,d),'z':(e,f)}
#     # --- optional base mass scaling (density scaling) ---
#     mass_scale_range: tuple[float, float] = (1.0, 1.0),             # scale base mass & inertia together
#     # --- optional *extra* COM jitter after taped-weight calc ---
#     extra_com_jitter_range: dict[str, tuple[float, float]] | None = None,  # small mm–cm jitter if desired
# ):
#     """
#     Update a single RigidObject to represent: original block + taped point mass.
#     Sets mass, COM, and inertia (about the NEW COM) consistently for the selected envs.

#     Args:
#         weight_mass_range: range to sample taped mass m_w [kg]. Use (0,0) to disable.
#         weight_offset_range: range to sample r (offset of taped mass from ORIGINAL COM, in body frame) [m].
#         mass_scale_range: density scaling for base body (scales base m and J_C linearly).
#         extra_com_jitter_range: optional small COM jitter *after* combining the taped mass (adjusts inertia with reverse PAT).
#     """
#     # --- helpers ---
#     def _sample_vec_range(rng_dict: dict[str, tuple[float, float]], E: int, device):
#         rng = torch.tensor([
#             rng_dict.get("x", (0.0, 0.0)),
#             rng_dict.get("y", (0.0, 0.0)),
#             rng_dict.get("z", (0.0, 0.0)),
#         ], dtype=torch.float32, device=device)
#         return math_utils.sample_uniform(rng[:, 0], rng[:, 1], (E, 3), device=device).to(torch.float32)

#     # --- setup ---
#     asset = env.scene[asset_cfg.name]
#     assert isinstance(asset, RigidObject), "This function is for RigidObjects only"

#     dev = torch.device("cpu")
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device=dev, dtype=torch.long)
#     else:
#         env_ids = env_ids.to(device=dev, dtype=torch.long)

#     # cache originals once (COM frame)
#     if not hasattr(asset, "_orig_massprops"):
#         asset._original_coms     = asset.root_physx_view.get_coms().clone().to(dev)      # (N,3)
#         asset._original_inertias = asset.root_physx_view.get_inertias().clone().to(dev)  # (N,9), row-major
#         asset._original_masses   = asset.root_physx_view.get_masses().clone().to(dev)    # (N,)
#         asset._orig_massprops = True

#     C0  = asset._original_coms
#     J0  = asset._original_inertias.view(-1, 3, 3)
#     m0  = asset._original_masses

#     E = len(env_ids)
#     eye3 = torch.eye(3, dtype=torch.float32, device=dev)

#     # --- sample base mass scaling (density scaling) ---
#     ms_min, ms_max = mass_scale_range
#     if ms_min == ms_max:
#         s = torch.full((E,), float(ms_min), device=dev)
#     else:
#         s = torch.empty(E, device=dev).uniform_(float(ms_min), float(ms_max))

#     # scaled base mass & inertia (geometry unchanged)
#     m_base = m0[env_ids] * s
#     J_base = (s.view(-1, 1, 1)) * J0[env_ids]

#     # --- sample taped weight mass ---
#     wm_min, wm_max = weight_mass_range
#     if wm_min == wm_max:
#         m_w = torch.full((E,), float(wm_min), device=dev)
#     else:
#         m_w = torch.empty(E, device=dev).uniform_(float(wm_min), float(wm_max))

#     # --- sample taped weight offset r (body frame, from ORIGINAL COM) ---
#     if weight_offset_range is None:
#         r = torch.zeros((E, 3), dtype=torch.float32, device=dev)
#     else:
#         r = _sample_vec_range(weight_offset_range, E, dev)

#     # --- combined mass and COM shift due to taped mass ---
#     m_prime = m_base + m_w                               # m' = m_base + m_w
#     # avoid divide-by-zero if both are zero (unlikely): clamp m'
#     m_prime = torch.where(m_prime <= 0.0, torch.full_like(m_prime, 1e-8), m_prime)

#     d = (m_w / m_prime).unsqueeze(1) * r                 # d = (m_w/m') * r
#     C_prime = C0[env_ids] + d                            # C' = C + d

#     # --- inertia about the NEW COM C' ---
#     # Block contribution (forward PAT from C to C'): J_block = J_base + m_base (||d||^2 I - d d^T)
#     J_block = torch.empty_like(J_base)
#     for k in range(E):
#         di = d[k]
#         di2 = torch.dot(di, di).item()
#         J_block[k] = J_base[k] + m_base[k].item() * (di2 * eye3 - torch.outer(di, di))

#     # Point-mass contribution at (C + r), expressed about C': J_point = m_w (||q||^2 I - q q^T), q = r - d
#     q = r - d
#     J_point = torch.empty_like(J_base)
#     for k in range(E):
#         qi = q[k]
#         qi2 = torch.dot(qi, qi).item()
#         J_point[k] = m_w[k].item() * (qi2 * eye3 - torch.outer(qi, qi))

#     J_prime = J_block + J_point

#     # --- optional extra COM jitter AFTER combination (small mm–cm) ---
#     if extra_com_jitter_range is not None:
#         d_extra = _sample_vec_range(extra_com_jitter_range, E, dev)  # Δd
#         C_prime = C_prime + d_extra
#         # reverse PAT to keep inertia about the new COM: J <- J - m' (||Δd||^2 I - Δd Δd^T)
#         for k in range(E):
#             de = d_extra[k]
#             de2 = torch.dot(de, de).item()
#             J_prime[k] = J_prime[k] - m_prime[k].item() * (de2 * eye3 - torch.outer(de, de))

#     # --- symmetrize (numerical hygiene) ---
#     J_prime = 0.5 * (J_prime + J_prime.transpose(1, 2))

#     # --- apply to selected envs only ---
#     asset.root_physx_view.set_masses(m_prime, env_ids)
#     asset.root_physx_view.set_coms(C_prime, env_ids)
#     asset.root_physx_view.set_inertias(J_prime.reshape(E, 9), env_ids)
# def _as_vec3(x: torch.Tensor) -> torch.Tensor:
#     # Accept (...,3) or (...,7) and return (...,3)
#     if x.shape[-1] == 3:
#         return x
#     if x.shape[-1] == 7:  # pose: [xyz, qx qy qz qw]
#         return x[..., :3]
#     raise RuntimeError(f"Expected last dim 3 or 7, got {tuple(x.shape)}")


# def randomize_rigid_body_com_tblock(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     asset_cfg: SceneEntityCfg,
#     *,
#     # --- taped weight (point mass) ---
#     weight_mass_range: tuple[float, float] = (0.0, 0.0),            # kg; include 0 to sometimes have no weight
#     weight_offset_range: dict[str, tuple[float, float]] | None = None,  # r from ORIGINAL COM (body frame), e.g., {"x":(-.07,.07),...}
#     # --- optional base mass scaling (density scaling) ---
#     mass_scale_range: tuple[float, float] = (1.0, 1.0),             # scale base m and J_C linearly
#     # --- optional *extra* COM jitter after taped-weight calc ---
#     extra_com_jitter_range: dict[str, tuple[float, float]] | None = None,  # small mm–cm jitter if desired
# ):
#     """
#     Single-object mass properties update to represent: original block + taped point mass.
#     Sets mass, COM (xyz), and inertia (about the NEW COM) consistently.
#     """
#     # ---- helpers ----------------------------------------------------------
#     def _sample_vec_range(rng_dict: dict[str, tuple[float, float]], E: int, device):
#         rng = torch.tensor([
#             rng_dict.get("x", (0.0, 0.0)),
#             rng_dict.get("y", (0.0, 0.0)),
#             rng_dict.get("z", (0.0, 0.0)),
#         ], dtype=torch.float32, device=device)  # (3,2)
#         return math_utils.sample_uniform(rng[:, 0], rng[:, 1], (E, 3), device=device).to(torch.float32)

#     def _resolve_body_index(asset, asset_cfg):
#         names = list(getattr(asset.root_physx_view, "body_names", []))
#         if not names:
#             return 0  # single-body view
#         bn = getattr(asset_cfg, "body_names", None)
#         name = bn if isinstance(bn, str) else (bn[0] if bn else None)
#         if name is None:
#             raise RuntimeError("This asset exposes multiple rigid bodies; pass a unique `body_names` in SceneEntityCfg.")
#         try:
#             return names.index(name)
#         except ValueError as e:
#             raise RuntimeError(f"Body name '{name}' not found. Available: {names}") from e

#     def _outer_batched(v: torch.Tensor) -> torch.Tensor:  # (E,3) -> (E,3,3)
#         v = v.unsqueeze(-1)  # (E,3,1)
#         return v @ v.transpose(-1, -2)

#     def _shift_term(v: torch.Tensor, mass: torch.Tensor, eye3: torch.Tensor) -> torch.Tensor:
#         # mass*(||v||^2 I - v v^T), with v: (E,3), mass: (E,)
#         n2 = (v * v).sum(dim=1, keepdim=True).unsqueeze(-1)  # (E,1,1)
#         return mass.view(-1, 1, 1) * (n2 * eye3 - _outer_batched(v))  # (E,3,3)

#     # ---- setup ------------------------------------------------------------
#     asset = env.scene[asset_cfg.name]
#     assert isinstance(asset, RigidObject), "This function is for RigidObjects only"

#     view = asset.root_physx_view
#     # Use PhysX tensors' device (usually CUDA), not CPU
#     dev = view.get_coms().device

#     # env_ids must be 1-D
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device=dev, dtype=torch.long)
#     else:
#         env_ids = env_ids.to(dev, torch.long).view(-1)
#     E = env_ids.numel()

#     # fetch full buffers and normalize to (N,B,*) once
#     coms_all    = view.get_coms().clone()       # (N,7) or (N,B,7) or (N,3)/(N,B,3)
#     inerts_all  = view.get_inertias().clone()   # (N,9) or (N,B,9)
#     masses_all  = view.get_masses().clone()     # (N,)  or (N,B) or (N,B,1)

#     if coms_all.ndim == 2:
#         coms_all   = coms_all.unsqueeze(1)                      # (N,1,7) or (N,1,3)
#         inerts_all = inerts_all.view(inerts_all.shape[0], 1, 9) # (N,1,9)
#         if masses_all.ndim == 1:
#             masses_all = masses_all.unsqueeze(1)                # (N,1)

#     N, B, last = coms_all.shape
#     pos_lastdim = 3
#     assert 0 <= B-1, "No bodies in view?"
#     body_idx = _resolve_body_index(asset, asset_cfg)
#     assert 0 <= body_idx < B, f"body_idx {body_idx} out of range for B={B}"

#     eye3 = torch.eye(3, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, 3, 3)

#     # gather selected envs (still useful for vectorized math)
#     coms_sel   = coms_all.index_select(0, env_ids).clone()      # (E,B,7) or (E,B,3)
#     inerts_sel = inerts_all.index_select(0, env_ids).clone()    # (E,B,9)
#     masses_sel = masses_all.index_select(0, env_ids).clone()    # (E,B) or (E,B,1)

#     # extract originals for the chosen body
#     C0 = _as_vec3(coms_sel[:, body_idx, :])                     # (E,3)
#     J0 = inerts_sel[:, body_idx, :].view(E, 3, 3)               # (E,3,3)
#     if masses_sel.ndim == 3:   # (E,B,1)
#         m0 = masses_sel[:, body_idx, 0].contiguous()            # (E,)
#     else:                      # (E,B)
#         m0 = masses_sel[:, body_idx].contiguous()               # (E,)

#     # ---- sample params ----------------------------------------------------
#     ms_min, ms_max = mass_scale_range
#     s = (torch.full((E,), float(ms_min), device=dev) if ms_min == ms_max
#          else torch.empty(E, device=dev).uniform_(float(ms_min), float(ms_max)))
#     m_base = m0 * s                           # (E,)
#     J_base = s.view(E, 1, 1) * J0             # (E,3,3)

#     wm_min, wm_max = weight_mass_range
#     m_w = (torch.full((E,), float(wm_min), device=dev) if wm_min == wm_max
#            else torch.empty(E, device=dev).uniform_(float(wm_min), float(wm_max)))

#     r = (torch.zeros((E, 3), dtype=torch.float32, device=dev) if weight_offset_range is None
#          else _sample_vec_range(weight_offset_range, E, dev))   # (E,3)

#     # ---- combine mass & COM ----------------------------------------------
#     m_prime = torch.clamp(m_base + m_w, min=1e-8)               # (E,)
#     d = (m_w / m_prime).unsqueeze(1) * r                        # (E,3)
#     C_prime = C0 + d                                            # (E,3)

#     # ---- inertia about NEW COM C' ----------------------------------------
#     J_block  = J_base  + _shift_term(d, m_base, eye3)           # (E,3,3)
#     q        = r - d
#     J_point  = _shift_term(q, m_w,    eye3)                     # (E,3,3)
#     J_prime  = 0.5 * (J_block + J_point + (J_block + J_point).transpose(1, 2))

#     if extra_com_jitter_range is not None:
#         d_extra = _sample_vec_range(extra_com_jitter_range, E, dev)    # (E,3)
#         C_prime = C_prime + d_extra
#         J_prime = J_prime - _shift_term(d_extra, m_prime, eye3)
#         J_prime = 0.5 * (J_prime + J_prime.transpose(1, 2))

#     # ---- write back (FULL buffers; no per-env set_*) ----------------------
#     # cast updates to view dtypes/devices
#     C_prime_w = C_prime.to(device=coms_all.device,   dtype=coms_all.dtype)         # (E,3)
#     J_prime_w = J_prime.reshape(E, 9).to(device=inerts_all.device, dtype=inerts_all.dtype)  # (E,9)
#     m_prime_w = m_prime.to(device=masses_all.device, dtype=masses_all.dtype)       # (E,)

#     # apply into full buffers at selected envs
#     coms_all[env_ids, body_idx, :pos_lastdim] = C_prime_w
#     inerts_all[env_ids, body_idx, :]          = J_prime_w
#     if masses_all.ndim == 3:   # (N,B,1)
#         masses_all[env_ids, body_idx, 0]      = m_prime_w
#     else:                       # (N,B)
#         masses_all[env_ids, body_idx]         = m_prime_w

#     # final invariants (cheap)
#     assert coms_all.ndim   == 3 and coms_all.shape[:2]   == (N, B)
#     assert inerts_all.ndim == 3 and inerts_all.shape[:2] == (N, B)
#     assert masses_all.ndim in (2,3) and masses_all.shape[:2] == (N, B)

#     # PhysX requires indices: provide all envs
#     indices_all = torch.arange(N, device=dev, dtype=torch.long)

#     view.set_masses   (masses_all.contiguous(),  indices_all)
#     view.set_coms     (coms_all.contiguous(),    indices_all)
#     view.set_inertias (inerts_all.contiguous(),  indices_all)

def _as_vec3(x: torch.Tensor) -> torch.Tensor:
    # Accept (...,3) or (...,7) and return (...,3)
    if x.shape[-1] == 3:
        return x
    if x.shape[-1] == 7:  # pose: [xyz, qx qy qz qw]
        return x[..., :3]
    raise RuntimeError(f"Expected last dim 3 or 7, got {tuple(x.shape)}")


import torch
from typing import Dict, Tuple, Optional

# Note: These imports need to be adjusted based on your actual module structure
# from your_module import ManagerBasedEnv, SceneEntityCfg, RigidObject
# from your_module import math_utils

import torch
from typing import Dict, Tuple, Optional

# Note: These imports need to be adjusted based on your actual module structure
# from your_module import ManagerBasedEnv, SceneEntityCfg, RigidObject


def sample_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample uniformly within a range.

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    # convert to tuple
    if isinstance(size, int):
        size = (size,)
    # return tensor
    return torch.rand(*size, device=device) * (upper - lower) + lower


def randomize_rigid_body_com_with_taped_weight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    *,
    # Taped weight parameters
    taped_weight_mass_range: tuple[float, float] = (0.0, 0.0),
    taped_weight_position_range: dict[str, tuple[float, float]] | None = None,  # Position relative to original COM
    # Base mass scaling (density scaling)
    base_mass_scale_range: tuple[float, float] = (1.0, 1.0),
    # Additional COM perturbation after weight attachment
    com_perturbation_range: dict[str, tuple[float, float]] | None = None,
):
    """
    Randomize rigid body mass properties by simulating a taped point mass.
    
    This function modifies a rigid body's mass properties as if a point mass (weight)
    was attached at a specified position. It preserves the original properties for
    future resets (no compounding across resets).
    
    Args:
        env: The environment containing the asset
        env_ids: Environment indices to randomize (None = all)
        asset_cfg: Configuration for the scene entity
        taped_weight_mass_range: Mass range for the attached weight [kg]
        taped_weight_position_range: Position of weight relative to original COM {axis: (min, max)}
        base_mass_scale_range: Scaling factor for the base object's mass (density scaling)
        com_perturbation_range: Additional COM offset after weight calculation {axis: (min, max)}
    
    The physics model:
        1. Scale the original body's mass (density scaling)
        2. Add a point mass at specified position relative to original COM
        3. Calculate new combined COM using weighted average
        4. Update inertia tensor for the new COM using parallel axis theorem
        5. Optionally add extra COM perturbation while preserving inertia
    """
    
    # ============== Helper Functions ==============
    
    def sample_3d_vector_from_ranges(
        ranges: dict[str, tuple[float, float]], 
        num_samples: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Sample 3D vectors with per-axis ranges."""
        range_tensor = torch.tensor([
            ranges.get("x", (0.0, 0.0)),
            ranges.get("y", (0.0, 0.0)),
            ranges.get("z", (0.0, 0.0)),
        ], dtype=torch.float32, device=device)
        
        # Validate ranges
        for axis, (min_val, max_val) in enumerate(range_tensor):
            if min_val > max_val:
                axis_name = ["x", "y", "z"][axis]
                raise ValueError(f"Invalid range for {axis_name}: min ({min_val}) > max ({max_val})")
        
        min_vals = range_tensor[:, 0]
        max_vals = range_tensor[:, 1]
        
        return math_utils.sample_uniform(
            min_vals, max_vals, (num_samples, 3), device=device
        ).to(torch.float32)
    
    def get_body_index_from_config(asset, asset_cfg) -> int:
        """Resolve body index from configuration."""
        body_names = list(getattr(asset.root_physx_view, "body_names", []))
        
        if not body_names:
            return 0  # Single body case
        
        # Extract body name from config
        config_body_names = getattr(asset_cfg, "body_names", None)
        if isinstance(config_body_names, str):
            target_name = config_body_names
        elif config_body_names:
            target_name = config_body_names[0]
        else:
            target_name = None
        
        if target_name is None:
            raise RuntimeError(
                "Multiple bodies detected but no body_names specified in SceneEntityCfg"
            )
        
        try:
            return body_names.index(target_name)
        except ValueError as e:
            raise RuntimeError(
                f"Body '{target_name}' not found. Available: {body_names}"
            ) from e
    
    def compute_outer_product_batch(vectors: torch.Tensor) -> torch.Tensor:
        """Compute outer product for batched vectors: v @ v^T."""
        vectors_col = vectors.unsqueeze(-1)  # (E,3) -> (E,3,1)
        return vectors_col @ vectors_col.transpose(-1, -2)  # (E,3,3)
    
    def compute_parallel_axis_shift(
        displacement: torch.Tensor, 
        mass: torch.Tensor, 
        identity_3x3: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute inertia shift using parallel axis theorem.
        Returns: mass * (||d||^2 * I - d @ d^T)
        """
        displacement_norm_sq = (displacement * displacement).sum(dim=1, keepdim=True).unsqueeze(-1)
        mass_expanded = mass.view(-1, 1, 1)
        
        return mass_expanded * (
            displacement_norm_sq * identity_3x3 - compute_outer_product_batch(displacement)
        )
    
    def ensure_vector3(tensor: torch.Tensor) -> torch.Tensor:
        """Extract xyz components, handling both 3D and 7D (with quaternion) formats."""
        if tensor.shape[-1] == 3:
            return tensor
        if tensor.shape[-1] == 7:  # pose: [xyz, qx qy qz qw]
            return tensor[..., :3]
        raise RuntimeError(f"Expected last dim 3 or 7, got {tuple(tensor.shape)}")
    
    # ============== Setup and Validation ==============
    
    asset = env.scene[asset_cfg.name]
    # Note: RigidObject should be imported from your module
    # assert isinstance(asset, RigidObject), "Asset must be a RigidObject"
    
    physx_view = asset.root_physx_view
    device = physx_view.get_coms().device
    
    # Prepare environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device, torch.long).view(-1)
    
    num_envs = env_ids.numel()
    
    # ============== Cache Original Properties ==============
    
    if not hasattr(asset, "_original_mass_properties_cached"):
        # Store original values on first call
        original_coms = physx_view.get_coms().clone()
        original_inertias = physx_view.get_inertias().clone()
        original_masses = physx_view.get_masses().clone()
        
        # Normalize to (N, B, *) format for consistent indexing
        if original_coms.ndim == 2:
            original_coms = original_coms.unsqueeze(1)
            original_inertias = original_inertias.view(original_inertias.shape[0], 1, 9)
            if original_masses.ndim == 1:
                original_masses = original_masses.unsqueeze(1)
        
        asset._original_coms = original_coms.to(device)
        asset._original_inertias = original_inertias.to(device)
        asset._original_masses = original_masses.to(device)
        asset._original_mass_properties_cached = True
    
    # ============== Extract Baseline and Current Values ==============
    
    baseline_coms = asset._original_coms
    baseline_inertias = asset._original_inertias
    baseline_masses = asset._original_masses
    
    # Get current values for shape compatibility
    current_coms = physx_view.get_coms().clone()
    current_inertias = physx_view.get_inertias().clone()
    current_masses = physx_view.get_masses().clone()
    
    # Normalize current values to (N, B, *) format
    if current_coms.ndim == 2:
        current_coms = current_coms.unsqueeze(1)
        current_inertias = current_inertias.view(current_inertias.shape[0], 1, 9)
        if current_masses.ndim == 1:
            current_masses = current_masses.unsqueeze(1)
    
    # Validate shape consistency
    if baseline_coms.shape != current_coms.shape:
        raise RuntimeError(
            f"Shape mismatch: baseline {baseline_coms.shape} vs current {current_coms.shape}"
        )
    
    body_idx = get_body_index_from_config(asset, asset_cfg)
    identity_3x3 = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, 3, 3)
    
    # ============== Extract Original Values for Selected Environments ==============
    
    original_com = ensure_vector3(
        baseline_coms.index_select(0, env_ids)[:, body_idx, :]
    )  # (E, 3)
    
    print(f"original_com: {original_com}")
    
    original_inertia = baseline_inertias.index_select(0, env_ids)[:, body_idx, :].view(
        num_envs, 3, 3
    )  # (E, 3, 3)
    
    masses_selected = baseline_masses.index_select(0, env_ids)
    if masses_selected.ndim == 3 and masses_selected.size(-1) == 1:
        original_mass = masses_selected[:, body_idx, 0]
    else:
        original_mass = masses_selected[:, body_idx]  # (E,)
    
    # ============== Sample Randomization Parameters ==============
    
    # Validate ranges
    min_scale, max_scale = base_mass_scale_range
    if min_scale > max_scale:
        raise ValueError(f"Invalid base_mass_scale_range: min ({min_scale}) > max ({max_scale})")
    if min_scale <= 0:
        raise ValueError(f"base_mass_scale_range min must be positive, got {min_scale}")
    
    min_weight, max_weight = taped_weight_mass_range
    if min_weight > max_weight:
        raise ValueError(f"Invalid taped_weight_mass_range: min ({min_weight}) > max ({max_weight})")
    if min_weight < 0:
        raise ValueError(f"taped_weight_mass_range min must be non-negative, got {min_weight}")
    
    # Sample base mass scaling
    if min_scale == max_scale:
        mass_scale = torch.full((num_envs,), float(min_scale), device=device)
    else:
        mass_scale = torch.empty(num_envs, device=device).uniform_(float(min_scale), float(max_scale))
    
    scaled_base_mass = original_mass * mass_scale
    scaled_base_inertia = mass_scale.view(num_envs, 1, 1) * original_inertia
    
    # Sample taped weight mass
    min_weight, max_weight = taped_weight_mass_range
    if min_weight == max_weight:
        taped_mass = torch.full((num_envs,), float(min_weight), device=device)
    else:
        taped_mass = torch.empty(num_envs, device=device).uniform_(float(min_weight), float(max_weight))
    
    # Sample taped weight position (relative to original COM)
    if taped_weight_position_range is None:
        weight_position = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    else:
        weight_position = sample_3d_vector_from_ranges(
            taped_weight_position_range, num_envs, device
        )
    
    # ============== Calculate New Mass Properties ==============
    
    # Total mass of combined system
    total_mass = torch.clamp(scaled_base_mass + taped_mass, min=1e-8)
    
    # COM shift due to added weight (weighted average formula)
    com_shift = (taped_mass / total_mass).unsqueeze(1) * weight_position
    new_com = original_com + com_shift
    
    # ============== Update Inertia Tensor for New COM ==============
    
    # Shift base object's inertia to new COM
    base_inertia_shifted = scaled_base_inertia + compute_parallel_axis_shift(
        com_shift, scaled_base_mass, identity_3x3
    )
    
    # Calculate point mass inertia about new COM
    weight_displacement_from_new_com = weight_position - com_shift
    point_mass_inertia = compute_parallel_axis_shift(
        weight_displacement_from_new_com, taped_mass, identity_3x3
    )
    
    # Combine inertias and ensure symmetry
    combined_inertia = base_inertia_shifted + point_mass_inertia
    new_inertia = 0.5 * (combined_inertia + combined_inertia.transpose(1, 2))
    
    # ============== Apply Optional COM Perturbation ==============
    
    if com_perturbation_range is not None:
        perturbation = sample_3d_vector_from_ranges(
            com_perturbation_range, num_envs, device
        )
        new_com = new_com + perturbation
        
        # Adjust inertia for COM perturbation (reverse parallel axis theorem)
        new_inertia = new_inertia - compute_parallel_axis_shift(
            perturbation, total_mass, identity_3x3
        )
        new_inertia = 0.5 * (new_inertia + new_inertia.transpose(1, 2))
    
    # ============== Write Back to Physics Engine ==============
    
    # Get fresh references to ensure correct device/dtype
    output_coms = physx_view.get_coms().clone()
    output_inertias = physx_view.get_inertias().clone()
    output_masses = physx_view.get_masses().clone()
    
    # Normalize output shapes if needed
    if output_coms.ndim == 2:
        output_coms = output_coms.unsqueeze(1)
        output_inertias = output_inertias.view(output_inertias.shape[0], 1, 9)
        if output_masses.ndim == 1:
            output_masses = output_masses.unsqueeze(1)
    
    # Cast to correct device/dtype and update
    new_com_cast = new_com.to(device=output_coms.device, dtype=output_coms.dtype)
    new_inertia_cast = new_inertia.reshape(num_envs, 9).to(
        device=output_inertias.device, dtype=output_inertias.dtype
    )
    total_mass_cast = total_mass.to(device=output_masses.device, dtype=output_masses.dtype)
    
    # Update values at selected environment indices
    output_coms[env_ids, body_idx, :3] = new_com_cast
    output_inertias[env_ids, body_idx, :] = new_inertia_cast
    
    if output_masses.ndim == 3 and output_masses.size(-1) == 1:
        output_masses[env_ids, body_idx, 0] = total_mass_cast
    else:
        output_masses[env_ids, body_idx] = total_mass_cast
    
    # Apply changes to physics engine (must pass all environment indices)
    all_indices = torch.arange(output_coms.shape[0], device=device, dtype=torch.long)
    physx_view.set_masses(output_masses.contiguous(), all_indices)
    physx_view.set_coms(output_coms.contiguous(), all_indices)
    physx_view.set_inertias(output_inertias.contiguous(), all_indices)

def randomize_rigid_body_collider_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    rest_offset_distribution_params: tuple[float, float] | None = None,
    contact_offset_distribution_params: tuple[float, float] | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the collider parameters of rigid bodies in an asset by adding, scaling, or setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect the collision checking.

    The function samples random values from the given distribution parameters and applies the operation to
    the collider properties. It then sets the values into the physics simulation. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.

    Currently, the distribution parameters are applied as absolute values.

    .. tip::
        This function uses CPU tensors to assign the collision properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")

    # sample collider properties from the given ranges and set into the physics simulation
    # -- rest offsets
    if rest_offset_distribution_params is not None:
        rest_offset = asset.root_physx_view.get_rest_offsets().clone()
        rest_offset = _randomize_prop_by_op(
            rest_offset,
            rest_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_rest_offsets(rest_offset, env_ids.cpu())
    # -- contact offsets
    if contact_offset_distribution_params is not None:
        contact_offset = asset.root_physx_view.get_contact_offsets().clone()
        contact_offset = _randomize_prop_by_op(
            contact_offset,
            contact_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_contact_offsets(contact_offset, env_ids.cpu())


def randomize_physics_scene_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gravity_distribution_params: tuple[list[float], list[float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize gravity by adding, scaling, or setting random values.

    This function allows randomizing gravity of the physics scene. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the
    operation.

    The distribution parameters are lists of two elements each, representing the lower and upper bounds of the
    distribution for the x, y, and z components of the gravity vector. The function samples random values for each
    component independently.

    .. attention::
        This function applied the same gravity for all the environments.

    .. tip::
        This function uses CPU tensors to assign gravity.
    """
    # get the current gravity
    gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
    dist_param_0 = torch.tensor(gravity_distribution_params[0], device="cpu")
    dist_param_1 = torch.tensor(gravity_distribution_params[1], device="cpu")
    gravity = _randomize_prop_by_op(
        gravity,
        (dist_param_0, dist_param_1),
        None,
        slice(None),
        operation=operation,
        distribution=distribution,
    )
    # unbatch the gravity tensor into a list
    gravity = gravity[0].tolist()

    # set the gravity into the physics simulation
    physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(*gravity))


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
        return _randomize_prop_by_op(
            data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
        )

    # Loop through actuators and randomize gains
    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            # we take all the joints of the actuator
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
        elif isinstance(actuator.joint_indices, slice):
            # we take the joints defined in the asset config
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            # we take the intersection of the actuator joints and the asset config joints
            actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            # the indices of the joints in the actuator that have to be randomized
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            # maps actuator indices that have to be randomized to global joint indices
            global_indices = actuator_joint_indices[actuator_indices]
        # Randomize stiffness
        if stiffness_distribution_params is not None:
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
            randomize(stiffness, stiffness_distribution_params)
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        # Randomize damping
        if damping_distribution_params is not None:
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
            randomize(damping, damping_distribution_params)
            actuator.damping[env_ids] = damping
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def randomize_joint_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # joint friction coefficient
    if friction_distribution_params is not None:
        friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_friction_coefficient_to_sim(
            friction_coeff[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
        )

    # joint armature
    if armature_distribution_params is not None:
        armature = _randomize_prop_by_op(
            asset.data.default_joint_armature.clone(),
            armature_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_armature_to_sim(armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids)

    # joint position limits
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        joint_pos_limits = asset.data.default_joint_pos_limits.clone()
        # -- randomize the lower limits
        if lower_limit_distribution_params is not None:
            joint_pos_limits[..., 0] = _randomize_prop_by_op(
                joint_pos_limits[..., 0],
                lower_limit_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
        # -- randomize the upper limits
        if upper_limit_distribution_params is not None:
            joint_pos_limits[..., 1] = _randomize_prop_by_op(
                joint_pos_limits[..., 1],
                upper_limit_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

        # extract the position limits for the concerned joints
        joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
        if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
            raise ValueError(
                "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater than"
                " upper joint limits. Please check the distribution parameters for the joint position limits."
            )
        # set the position limits into the physics simulation
        asset.write_joint_position_limit_to_sim(
            joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
        )


def randomize_fixed_tendon_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    limit_stiffness_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    rest_length_distribution_params: tuple[float, float] | None = None,
    offset_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the simulated fixed tendon parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the fixed tendon parameters of the asset.
    These correspond to the physics engine tendon properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to the tendon properties.
    It then sets the values into the physics simulation. If the distribution parameters are not provided for a
    particular property, the function does not modify the property.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.fixed_tendon_ids == slice(None):
        tendon_ids = slice(None)  # for optimization purposes
    else:
        tendon_ids = torch.tensor(asset_cfg.fixed_tendon_ids, dtype=torch.int, device=asset.device)

    # sample tendon properties from the given ranges and set into the physics simulation
    # stiffness
    if stiffness_distribution_params is not None:
        stiffness = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_stiffness.clone(),
            stiffness_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_stiffness(stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # damping
    if damping_distribution_params is not None:
        damping = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_damping.clone(),
            damping_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_damping(damping[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # limit stiffness
    if limit_stiffness_distribution_params is not None:
        limit_stiffness = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_limit_stiffness.clone(),
            limit_stiffness_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_limit_stiffness(limit_stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # position limits
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        limit = asset.data.default_fixed_tendon_pos_limits.clone()
        # -- lower limit
        if lower_limit_distribution_params is not None:
            limit[..., 0] = _randomize_prop_by_op(
                limit[..., 0],
                lower_limit_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
        # -- upper limit
        if upper_limit_distribution_params is not None:
            limit[..., 1] = _randomize_prop_by_op(
                limit[..., 1],
                upper_limit_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )

        # check if the limits are valid
        tendon_limits = limit[env_ids[:, None], tendon_ids]
        if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are greater"
                " than upper tendon limits."
            )
        asset.set_fixed_tendon_position_limit(tendon_limits, tendon_ids, env_ids)

    # rest length
    if rest_length_distribution_params is not None:
        rest_length = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_rest_length.clone(),
            rest_length_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_rest_length(rest_length[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # offset
    if offset_distribution_params is not None:
        offset = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_offset.clone(),
            offset_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_offset(offset[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # write the fixed tendon properties into the simulation
    asset.write_fixed_tendon_properties_to_sim(tendon_ids, env_ids)


def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )


def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset nodal state to a random position and velocity uniformly within the given ranges.

    This function randomizes the nodal position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default nodal position, before setting
      them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis. The keys of the
    dictionary are ``x``, ``y``, ``z``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: DeformableObject = env.scene[asset_cfg.name]
    # get default root state
    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()

    # position
    range_list = [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., :3] += rand_samples

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., 3:] += rand_samples

    # set into the physics simulation
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    If :attr:`reset_joint_targets` is True, the joint position and velocity targets of the articulations are
    also reset to their default values. This might be useful for some cases to clear out any previously set targets.
    However, this is not the default behavior as based on our experience, it is not always desired to reset
    targets to default values, especially when the targets should be handled by action terms and not event terms.
    """
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        # reset joint targets if required
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


class randomize_visual_texture_material(ManagerTermBase):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual texture material with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # join all bodies in the asset
        body_names = asset_cfg.body_names
        if isinstance(body_names, str):
            body_names_regex = body_names
        elif isinstance(body_names, list):
            body_names_regex = "|".join(body_names)
        else:
            body_names_regex = ".*"

        # create the affected prim path
        # Check if the pattern with '/visuals' yields results when matching `body_names_regex`.
        # If not, fall back to a broader pattern without '/visuals'.
        asset_main_prim_path = asset.cfg.prim_path
        pattern_with_visuals = f"{asset_main_prim_path}/{body_names_regex}/visuals"
        # Use sim_utils to check if any prims currently match this pattern
        matching_prims = sim_utils.find_matching_prim_paths(pattern_with_visuals)
        if matching_prims:
            # If matches are found, use the pattern with /visuals
            prim_path = pattern_with_visuals
        else:
            # If no matches found, fall back to the broader pattern without /visuals
            # This pattern (e.g., /World/envs/env_.*/Table/.*) should match visual prims
            # whether they end in /visuals or have other structures.
            prim_path = f"{asset_main_prim_path}/.*"
            carb.log_info(
                f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{prim_path}' for texture"
                " randomization."
            )

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            texture_paths = cfg.params.get("texture_paths")
            event_name = cfg.params.get("event_name")
            texture_rotation = cfg.params.get("texture_rotation", (0.0, 0.0))

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            # Create the omni-graph node for the randomization term
            def rep_texture_randomization():
                prims_group = rep.get.prims(path_pattern=prim_path)

                with prims_group:
                    rep.randomizer.texture(
                        textures=texture_paths,
                        project_uvw=True,
                        texture_rotate=rep.distribution.uniform(*texture_rotation),
                    )
                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_texture_randomization()
        else:
            # acquire stage
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=prim_path, stage=stage)

            num_prims = len(prims_group)
            # rng that randomizes the texture and rotation
            self.texture_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            # read parameters from the configuration
            texture_paths = texture_paths if texture_paths else self._cfg.params.get("texture_paths")
            texture_rotation = (
                texture_rotation if texture_rotation else self._cfg.params.get("texture_rotation", (0.0, 0.0))
            )

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            num_prims = len(self.material_prims)
            random_textures = self.texture_rng.generator.choice(texture_paths, size=num_prims)
            random_rotations = self.texture_rng.generator.uniform(
                texture_rotation[0], texture_rotation[1], size=num_prims
            )

            # modify the material properties
            rep.functional.modify.attribute(self.material_prims, "diffuse_texture", random_textures)
            rep.functional.modify.attribute(self.material_prims, "texture_rotate", random_rotations)


class randomize_visual_color(ManagerTermBase):
    """Randomize the visual color of bodies on an asset using Replicator API.

    This function randomizes the visual color of the bodies of the asset using the Replicator API.
    The function samples random colors from the given colors and applies them to the bodies
    of the asset.

    The function assumes that the asset follows the prim naming convention as:
    "{asset_prim_path}/{mesh_name}" where the mesh name is the name of the mesh to
    which the color is applied. For instance, if the asset has a prim path "/World/asset"
    and a mesh named "body_0/mesh", the prim path for the mesh would be
    "/World/asset/body_0/mesh".

    The colors can be specified as a list of tuples of the form ``(r, g, b)`` or as a dictionary
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    If a dictionary is used, the function will sample random colors from the given ranges.

    .. note::
        When randomizing the color of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        mesh_name: str = cfg.params.get("mesh_name", "")  # type: ignore

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual color with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # create the affected prim path
        if not mesh_name.startswith("/"):
            mesh_name = "/" + mesh_name
        mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        # TODO: Need to make it work for multiple meshes.

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            colors = cfg.params.get("colors")
            event_name = cfg.params.get("event_name")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = rep.distribution.uniform(color_low, color_high)
            else:
                colors = list(colors)

            # Create the omni-graph node for the randomization term
            def rep_color_randomization():
                prims_group = rep.get.prims(path_pattern=mesh_prim_path)
                with prims_group:
                    rep.randomizer.color(colors=colors)

                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_color_randomization()
        else:
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=mesh_prim_path, stage=stage)

            num_prims = len(prims_group)
            self.color_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            colors = colors if colors else self._cfg.params.get("colors")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = [color_low, color_high]
            else:
                colors = list(colors)

            num_prims = len(self.material_prims)
            random_colors = self.color_rng.generator.uniform(colors[0], colors[1], size=(num_prims, 3))

            rep.functional.modify.attribute(self.material_prims, "diffuse_color_constant", random_colors)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data
