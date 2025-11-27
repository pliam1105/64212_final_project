import os
import trimesh


def convert_stl_to_obj(input_directory):
    # Get a list of all files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".stl"):  # Check if it's an STL file
            stl_path = os.path.join(input_directory, filename)
            obj_path = os.path.join(input_directory, filename.replace(".stl", ".obj"))

            # Load the STL file using trimesh
            try:
                mesh = trimesh.load_mesh(stl_path)

                # Export to OBJ format
                mesh.export(obj_path)
                print(f"Successfully converted: {filename} to {filename.replace('.stl', '.obj')}")

            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


if __name__ == "__main__":
    # Set the directory where your STL files are located
    input_directory = "."

    # Call the conversion function
    convert_stl_to_obj(input_directory)
