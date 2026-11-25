"""Build the cuRobo robot model of the SO-101 (runs once on the GPU server, cuRobo v2 interpreter).

Input: the packaged SO-101 geometry archive (``geometry_model.zip`` + ``physical.json`` from
``ni_rua/robot/model/robot_information_data/so101``, copied next to this script by ``deploy.sh``) and the deployment's
joint-limit margin. Output directory ``--out``: ``so101/`` (URDF + STL meshes), ``so101_margin.urdf`` (the five arm
joint limits shrunk by ``--margin`` rad on both sides, the limits NI-RUA plans with), ``so101_curobo.yml`` (cuRobo
kinematics with fitted collision spheres and the self-collision ignore matrix; the jaw joint locked at 0 rad) and
``model.json`` (margin, joint limits in rad, URDF sha256, sphere count).

    python build_model.py --source model_src --out model --margin 0.05

Collision spheres are fitted by cuRobo's ``RobotBuilder`` (MorphIt); the base link spheres are clipped at its z=0 so
they do not reach below the mounting surface.
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def shrink_limits(urdf_text, margin):
    """URDF text with each arm joint's ``lower``/``upper`` moved ``margin`` rad inward; returns (text, limits)."""
    limits = {}

    def joint(match):
        block = match.group(0)
        name = re.search(r'name="([^"]+)"', block).group(1)
        if name not in ARM_JOINTS:
            return block
        found = re.search(r'lower="([-0-9.eE+]+)" upper="([-0-9.eE+]+)"', block)
        lower, upper = float(found.group(1)) + margin, float(found.group(2)) - margin
        if lower >= upper:
            raise SystemExit(f"margin {margin} rad leaves no range on {name}")
        limits[name] = [lower, upper]
        return block[:found.start()] + f'lower="{lower!r}" upper="{upper!r}"' + block[found.end():]

    text = re.sub(r'<joint name="[^"]+" type="revolute">.*?</joint>', joint, urdf_text, flags=re.S)
    if set(limits) != set(ARM_JOINTS):
        raise SystemExit(f"URDF lacks arm joints: {sorted(set(ARM_JOINTS) - set(limits))}")
    return text, limits


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="directory with geometry_model.zip and physical.json")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--margin", type=float, required=True, help="joint-limit margin per side, rad")
    parser.add_argument("--seed", type=int, default=42, help="NumPy / PyTorch RNG seed of the sphere fit (default 42)")
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.out)
    native = json.loads((source / "physical.json").read_text(encoding="utf-8"))["geometry"]["native_model"]
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source / f"{native['resource_id']}.zip") as archive:
        archive.extractall(out / "so101")
    original = (out / "so101" / native["assembly_member"]).read_text(encoding="utf-8")
    text, limits = shrink_limits(original, args.margin)
    urdf = out / "so101" / "so101_margin.urdf"
    urdf.write_text(text, encoding="utf-8")

    import numpy as np
    import torch
    import yaml
    from curobo.robot_builder import RobotBuilder

    # MorphIt sphere fitting and collision sampling are stochastic; pin the RNGs so a rebuild gives the same model.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    builder = RobotBuilder(urdf_path=str(urdf), asset_path=str(out / "so101"), tool_frames=["gripper_frame_link"])
    builder.fit_collision_spheres(sphere_density=1.0, compute_metrics=False, clip_links={"base_link": ("z", 0.0)})
    builder.compute_collision_matrix(prune_collisions=True, num_samples=2000)
    yml = out / "so101_curobo.yml"
    builder.save(builder.build(), str(yml))
    document = yaml.safe_load(yml.read_text(encoding="utf-8"))
    kinematics = document["kinematics"]
    kinematics["lock_joints"] = {"gripper": 0.0}
    # Parts mounted on gripper_link right next to wrist_link: the builder already ignores wrist_link against the camera
    # housing; the thin-shell camera mount gets a poor sphere fit (under 1 % coverage) whose spheres reach into
    # wrist_link in pointing-down poses the real arm reaches (commissioned home pose), so ignore that pair as well.
    ignore = kinematics.setdefault("self_collision_ignore", {})
    for link, other in (("wrist_link", "wrist_camera_mount_link"),):
        ignore.setdefault(link, [])
        if other not in ignore[link]:
            ignore[link].append(other)
    yml.write_text(yaml.safe_dump(document), encoding="utf-8")
    info = dict(margin_rad=args.margin, seed=args.seed, joint_limits_rad=limits, tool_frame="gripper_frame_link",
                urdf_sha256=hashlib.sha256(original.encode()).hexdigest(), spheres=int(builder.num_spheres))
    (out / "model.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
