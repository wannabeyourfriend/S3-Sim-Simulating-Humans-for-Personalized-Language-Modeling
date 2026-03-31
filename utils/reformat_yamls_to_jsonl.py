import os
import json
import yaml
from glob import glob

def yaml_files_to_jsonl(folder_path: str, output_file: str = "output.jsonl") -> None:
    """
    Convert all YAML files in the specified folder into a single JSONL file.
    
    Args:
        folder_path (str): Path to the folder containing YAML files.
        output_file (str): Name of the output JSONL file.
    """
    yaml_files = glob(os.path.join(folder_path, "*.yaml")) + glob(os.path.join(folder_path, "*.yml"))
    
    with open(output_file, "w", encoding="utf-8") as outf:
        for yaml_file in yaml_files:
            with open(yaml_file, "r", encoding="utf-8") as inf:
                try:
                    data = yaml.safe_load(inf)
                    json_line = json.dumps(data, ensure_ascii=False)
                    outf.write(json_line + "\n")
                except yaml.YAMLError as e:
                    print(f"Error parsing {yaml_file}: {e}")

if __name__ == "__main__":
    folder_path = "data/behavior_modes/yaml"
    output_file = "data/behavior_modes/behavior_modes.jsonl"
    yaml_files_to_jsonl(folder_path, output_file)


# Example usage:
# yaml_files_to_jsonl("path/to/your/folder", "combined.jsonl")
