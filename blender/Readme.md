# Dataset generator utility

Here you can find a blender script that allows to generate random scenes and render both the depth and the normal maps.

## Usage

1. Open Blender (tested with version 3.0.1)
2. Go to the Scripting tab, and import the [blender script](/blender/blender_script.py)
3. Update the [Normal Depth scene](/blender/normal_depth_scene.fbx) path, and change the resolution/format/loops as you prefer
4. Save the scene
5. Run the script. Output data will be in the same folder as the blender scene

## GPU

To use the GPU for rendering, remember to enable it in Edit->Preferences->System.