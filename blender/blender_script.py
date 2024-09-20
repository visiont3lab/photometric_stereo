import bpy
import random
import math
import os
import mathutils

max_loop = 1
num_objects = 10
file_format = 'BMP'
override_material_path = "/home/andrea/Desktop/gitproj/photometric_stereo/blender/normal_depth_scene.fbx"
resolution = 512


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def get_override_material_from_fbx(fbx_file_path):
    clear_scene()
    bpy.ops.import_scene.fbx(filepath=fbx_file_path)
    return bpy.data.objects[0].active_material

def set_cycles_gpu_rendering():
    # Set render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    # Set device to GPU
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    # Enable all available GPUs
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons['cycles'].preferences
    cycles_preferences.compute_device_type = 'CUDA'  # or 'OPTIX' for NVIDIA RTX cards, or 'HIP' for AMD
    
    for device in cycles_preferences.devices:
        device.use = True
    
def load_fbx_and_set_material_override(fbx_path, object_name=None):
    # Load the FBX file
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    # Get the imported objects
    imported_objects = bpy.context.selected_objects
    
    if not imported_objects:
        return None
    
    # If object_name is specified, try to find that object
    if object_name:
        target_object = next((obj for obj in imported_objects if obj.name == object_name), None)
        if not target_object:
            return None
    else:
        # If no object_name specified, use the first object with a material
        target_object = next((obj for obj in imported_objects if obj.material_slots), None)
        if not target_object:
            return None
    
    # Get the material from the object
    if not target_object.material_slots:
        return None
    
    override_material = target_object.material_slots[0].material
    
    # Set the material as override for all view layers
    for view_layer in bpy.context.scene.view_layers:
        view_layer.material_override = override_material
    return override_material

def create_random_object():
    obj_type = random.choice(['CUBE', 'SPHERE', 'CONE', 'CYLINDER'])
    if obj_type == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(size=random.uniform(0.5, 2.0))
    elif obj_type == 'SPHERE':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.5, 2.0))
    elif obj_type == 'CONE':
        bpy.ops.mesh.primitive_cone_add(radius1=random.uniform(0.5, 2.0), depth=random.uniform(1.0, 3.0))
    elif obj_type == 'CYLINDER':
        bpy.ops.mesh.primitive_cylinder_add(radius=random.uniform(0.5, 2.0), depth=random.uniform(1.0, 3.0))
    
    obj = bpy.context.active_object
    obj.location = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 5))

def add_fixed_plane():
    bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False, location=(0, 0, 0))

def setup_camera(random_spawn=False):
    if random_spawn:
        # Create a new camera
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        # Set random position
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        z = random.uniform(5, 10)
        camera.location = (x, y, z)

        # Point the camera to the origin (0, 0, 0)
        look_at = mathutils.Vector((0, 0, 0))
        direction = look_at - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
    else:
        bpy.ops.object.camera_add(location=(0, 0, 10))
        camera = bpy.context.active_object

    # Set as active camera
    bpy.context.scene.camera = camera

def setup_compositing():
    # Clear existing nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create nodes
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    composite_normal = tree.nodes.new(type='CompositorNodeComposite')
    composite_depth = tree.nodes.new(type='CompositorNodeComposite')
    invert = tree.nodes.new(type='CompositorNodeInvert')
    normalize = tree.nodes.new(type='CompositorNodeNormalize')
    file_output_normal = tree.nodes.new(type='CompositorNodeOutputFile')
    file_output_depth = tree.nodes.new(type='CompositorNodeOutputFile')

    # Set up node properties
    invert.invert_rgb = True
    invert.invert_alpha = False
    file_output_normal.base_path = "//dataset/normals"
    file_output_depth.base_path = "//dataset/depths"
    file_output_normal.file_slots[0].use_node_format = False
    file_output_normal.file_slots[0].format.file_format = file_format
    file_output_depth.file_slots[0].use_node_format = False
    file_output_depth.file_slots[0].format.file_format = file_format

    # Create links
    links = tree.links
    links.new(render_layers.outputs['Image'], composite_normal.inputs['Image'])
    links.new(render_layers.outputs['Image'], file_output_normal.inputs['Image'])
    links.new(render_layers.outputs['Alpha'], composite_depth.inputs['Alpha'])
    links.new(render_layers.outputs['Depth'], invert.inputs['Color'])
    links.new(invert.outputs['Color'], normalize.inputs['Value'])
    links.new(normalize.outputs['Value'], file_output_depth.inputs['Image'])
    links.new(normalize.outputs['Value'], composite_depth.inputs['Image'])
    links.new(render_layers.outputs['Alpha'], composite_depth.inputs['Alpha'])
    render_layers.scene.view_layers["ViewLayer"].material_override = get_override_material_from_fbx(override_material_path)
    # Get the current view layer
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True

    # Output normal map - Adjust normals to 0-1 range
    normal_map_node = tree.nodes.new(type='CompositorNodeMixRGB')
    normal_map_node.blend_type = 'ADD'
    normal_map_node.inputs[1].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], normal_map_node.inputs[2])
    links.new(normal_map_node.outputs['Image'], file_output_normal.inputs['Image'])

def main():
    set_cycles_gpu_rendering()
    setup_compositing()
    for i in range (max_loop):
        clear_scene()
        for j in range(num_objects):
            create_random_object()
        add_fixed_plane()
        setup_camera()
        bpy.ops.render.render()
        bpy.context.scene.frame_set(i)

if __name__ == "__main__":
    main()