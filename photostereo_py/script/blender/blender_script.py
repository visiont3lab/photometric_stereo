import bpy
import random
import os

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
    print(f"Created {obj_type} at {obj.location}")

def setup_camera():
    bpy.ops.object.camera_add(location=(0, -10, 5))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0)
    bpy.context.scene.camera = camera
    print(f"Camera set up at {camera.location}")

def render_normal_map():
    print("Setting up normal map rendering")
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = 0, 0
    
    normal_map = tree.nodes.new('CompositorNodeNormalize')  # Changed to Normalize node
    normal_map.location = 200, 0
    
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.location = 400, 0
    output.base_path = "//normal_maps/"
    output.file_slots[0].path = "normal_map_"
    output.format.file_format = 'BMP'
    
    links.new(rl.outputs['Normal'], normal_map.inputs[0])
    links.new(normal_map.outputs[0], output.inputs[0])
    
    output_path = bpy.path.abspath(output.base_path)
    os.makedirs(output_path, exist_ok=True)
    
    bpy.context.scene.render.filepath = os.path.join(output_path, "normal_map")
    bpy.ops.render.render(write_still=True)

def render_depth_map():
    print("Setting up depth map rendering")
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = 0, 0
    
    map_value = tree.nodes.new('CompositorNodeMapRange')  # Changed to MapRange node
    map_value.location = 200, 0
    map_value.inputs[1].default_value = 0  # Min
    map_value.inputs[2].default_value = 10  # Max
    
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.location = 400, 0
    output.base_path = "//depth_maps/"
    output.file_slots[0].path = "depth_map_"
    output.format.file_format = 'BMP'
    
    links.new(rl.outputs['Depth'], map_value.inputs[0])
    links.new(map_value.outputs[0], output.inputs[0])
    
    output_path = bpy.path.abspath(output.base_path)
    os.makedirs(output_path, exist_ok=True)
    print(f"Depth map will be saved to: {output_path}")
    
    bpy.context.scene.render.filepath = os.path.join(output_path, "depth_map")
    bpy.ops.render.render(write_still=True)
    print("Depth map rendering complete")

def main():
    print("Starting main function")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    for i in range(10):
        create_random_object()
        print(f"Created object {i+1}")
    
    setup_camera()
    
    render_normal_map()
    render_depth_map()
    
    print("Script execution complete")

if __name__ == "__main__":
    main()