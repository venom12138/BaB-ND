from pathlib import Path
import os

import numpy as np
import sapien.core as sapien

def load_open_box(scene: sapien.Scene, renderer: sapien.SapienRenderer, half_l, half_w, h, floor_width, origin=[0, 0, 0], color=(161./225., 102./225., 47./225.)):
    asset_dir = Path(__file__).parent.parent / "assets"
    map_path = asset_dir / "misc" / "dark-wood.png"
    box_visual_material = renderer.create_material()
    box_visual_material.set_metallic(0.0)
    box_visual_material.set_specular(0.3)
    box_visual_material.set_diffuse_texture_from_file(str(map_path))
    box_visual_material.set_roughness(0.3)
    
    builder = scene.create_actor_builder()
    
    # bottom
    box_bottom_origin = origin.copy()
    box_bottom_origin[2] += floor_width/2
    builder.add_box_collision(half_size=[half_l, half_w, floor_width/2], pose=sapien.Pose(p = box_bottom_origin))
    builder.add_box_visual(half_size=[half_l, half_w, floor_width/2], material=box_visual_material, pose=sapien.Pose(p = box_bottom_origin))
    
    # left
    box_left_origin = origin.copy()
    box_left_origin[0] += -half_l
    box_left_origin[2] += h/2
    builder.add_box_collision(half_size=[floor_width/2, half_w, h/2], pose=sapien.Pose(p = box_left_origin))
    builder.add_box_visual(half_size=[floor_width/2, half_w, h/2], material=box_visual_material, pose=sapien.Pose(p = box_left_origin))
    
    # right
    box_right_origin = origin.copy()
    box_right_origin[0] += half_l
    box_right_origin[2] += h/2
    builder.add_box_collision(half_size=[floor_width/2, half_w, h/2], pose=sapien.Pose(p = box_right_origin))
    builder.add_box_visual(half_size=[floor_width/2, half_w, h/2], material=box_visual_material, pose=sapien.Pose(p = box_right_origin))
    
    # back
    box_back_origin = origin.copy()
    box_back_origin[1] += -half_w
    box_back_origin[2] += h/2
    builder.add_box_collision(half_size=[half_l, floor_width/2, h/2], pose=sapien.Pose(p = box_back_origin))
    builder.add_box_visual(half_size=[half_l, floor_width/2, h/2], material=box_visual_material, pose=sapien.Pose(p = box_back_origin))
    
    # front
    box_front_origin = origin.copy()
    box_front_origin[1] += half_w
    box_front_origin[2] += h/2
    builder.add_box_collision(half_size=[half_l, floor_width/2, h/2], pose=sapien.Pose(p = box_front_origin))
    builder.add_box_visual(half_size=[half_l, floor_width/2, h/2], material=box_visual_material, pose=sapien.Pose(p = box_front_origin))
    
    box = builder.build_static(name='open box')
    return box

YX_DEFAULT_SCALE = {
    # 'diet_soda': [0.008, 0.012, 0.008],
    'cola': 0.5,
    'sprite': 0.5,
    'pepsi': 2.0,
    'mtndew': 0.03,
    'obamna': 0.0003,
    'diet_soda': 0.008,
    'drpepper': 0.04,
    'white_pepsi': 0.016,
    'mug_tree': 1.1,
    'nescafe_mug': 0.013,
}

def load_yx_obj(scene: sapien.Scene, object_name, scale=None, material=None, collision_shape = 'convex', density=1000, is_static=False):
    current_dir = Path(__file__).parent
    yx_dir = current_dir.parent / "assets" / "yx"
    
    visual_file = yx_dir / object_name /  f"{object_name}.glb"
    collision_file_cands = [
        yx_dir / object_name / f"decomp.obj",
        yx_dir / object_name / f"{object_name}_collision.obj",
        yx_dir / object_name / f"{object_name}.glb",
    ]
    
    builder = scene.create_actor_builder()
    
    if scale is None:
        scale = YX_DEFAULT_SCALE[object_name]
    if isinstance(scale, float):
        scales = np.array([scale] * 3)
    elif isinstance(scale, list):
        scales = np.array(scale)
    else:
        print("scale must be float or list")
        raise NotImplementedError
    builder.add_visual_from_file(str(visual_file), scale=scales)
    for collision_file in collision_file_cands:
        if os.path.exists(str(collision_file)):
            if collision_shape == 'convex':
                builder.add_collision_from_file(str(collision_file), scale=scales, material=material, density=density)
            elif collision_shape == 'nonconvex':
                builder.add_nonconvex_collision_from_file(str(collision_file), scale=scales, material=material, density=density)
            elif collision_shape == 'multiple':
                builder.add_multiple_collisions_from_file(str(collision_file), scale=scales, material=material, density=density)
            break
    
    if is_static:
        actor = builder.build_static(name=object_name)
    else:
        actor = builder.build(name=object_name)
    return actor
