import os
import sys
import pathlib
import glob

# Example file:
# equiformer/scripts/train/md17/equiformer/se_l2/target@aspirin.sh

# all files recursively in equiformer/scripts/train are to be converted to hydra
# all_files = glob.glob('equiformer/scripts/train/**/*.sh', recursive=True)
all_files = glob.glob("equiformer/scripts/train/md17/**/*.sh", recursive=True)

# at the beginning of new file
default_beginning = """# @package _global_
# ^^^ this @package directive solves any nesting problem (if this file is included in another folder)
"""

dest_dir = "equiformer/config/preset"

# some keys are passed to the model:
# irreps_in=args.input_irreps,
# radius=args.radius,
# number_of_basis=args.number_of_basis,
# task_mean=mean,
# task_std=std,
# atomref=None,
# path_drop=args.path_drop
model_keys = [
    "irreps_in",
    "radius",
    "number_of_basis",
    "atomref",
    "path_drop",
]
keys_to_ignore = ["task_mean", "task_std"]
keys_to_replace = {
    "input_irreps": "irreps_in",
    "radius": "max_radius",
    "num_basis": "number_of_basis",
}

for file in all_files:
    if "md17" in file:
        dataset = file.split("/")[3]
        equivariance = file.split("/")[5]
        equivariance = equivariance.replace("_", "")

        # get the path to the file
        path = pathlib.Path(file)
        # get the path to the file without the file name
        path = path.parent
        # get the file name
        file_name = os.path.basename(file)
        # get the name of the file without the extension
        file_name = os.path.splitext(file_name)[0]
        # get the target
        target = file_name.split("@")[1]

        # get the path to the new file
        new_file = f"{dest_dir}/{dataset}_{target}_{equivariance}.yaml"

        # open the new file
        with open(new_file, "w") as f:
            # write the hydra config
            f.write(f"")

            f.write(default_beginning)
            f.write("\n")
            f.write(f"# auto-generated from {file}")
            f.write("\n")

            # read the file
            model_name = None
            keys = []
            model = {}
            with open(file, "r") as oldf:
                lines = oldf.readlines()

                for line in lines:
                    if "--" in line:
                        # --output-dir 'models/qm9/equiformer/se_l2/target@0/' \
                        # line starts after '--'
                        line = line.split("--")[1]
                        # remove the last character if it is a '\'
                        if line[-1] == "\n":
                            line = line[:-1]
                        if line[-1] == "\\":
                            line = line[:-1]
                        # replace the first space with ':'
                        line = line.replace(" ", ":", 1)
                        line = line.replace(" ", "")
                        # if the first argument contains '-' replace it with '_'
                        key = line.split(":")[0]
                        key = key.replace("-", "_")
                        if line.split(":") == 1:
                            value = "True"
                        else:
                            value = line.split(":")[1]

                        for k, v in keys_to_replace.items():
                            if key == k:
                                key = key.replace(k, v)

                        for k in keys_to_ignore:
                            if key == k:
                                continue

                        if key in model_keys:
                            model[key] = value
                        else:
                            # write the line
                            f.write(f"{key}: {value}\n")

                        keys.append(key)
                        if key in ["model_name", "model", "name"]:
                            model_name = value

            model_name = model_name.replace("-", "_").replace("'", "")
            # print(' model_name:', model_name)
            # print(' keys:', keys)

            # two new lines
            f.write("\n\n")

            # look for model config
            # print(' Looking for:', f'def {model_name}')
            configs = [
                "equiformer/nets/dp_attention_transformer_md17.py",
                "equiformer/nets/graph_attention_transformer_md17.py",
            ]
            for cfg in configs:
                # open file
                with open(cfg, "r") as f_cfg:
                    cfg_lines = f_cfg.readlines()

                    start_recording = False
                    for cfgline in cfg_lines:
                        # if line starts with 'model_name'
                        if cfgline.startswith(f"def {model_name}"):
                            start_recording = True
                            # print(f' Found {model_name} in {cfg}')
                            f.write(f"# auto-generated from {cfg}\n")
                            f.write(f"model:\n")

                        elif start_recording:
                            if cfgline.startswith("):") or cfgline.startswith(
                                "**kwargs"
                            ):
                                break

                            cfgline = cfgline.replace(" ", "")
                            cfgline = cfgline.replace("=", ": ")

                            # if the key is already in the .sh file, it is overwritten anyways
                            key = cfgline.split(":")[0]
                            if key in keys:
                                continue

                            # if there is no value, it is a required argument
                            # skip
                            if ":" not in cfgline:
                                continue

                            # replace None with null
                            cfgline = cfgline.replace("None", "null")

                            if cfgline[-1] == "\n":
                                cfgline = cfgline[:-1]
                            if cfgline[-1] == ",":
                                cfgline = cfgline[:-1]
                            f.write(f"  {cfgline}\n")

                    if start_recording:
                        # finally write the model
                        if len(model) > 0:
                            # f.write('model:\n')
                            for key, value in model.items():
                                f.write(f"  {key}: {value}\n")
                        break
            if not start_recording:
                print(f"Could not find {model_name} in any of the configs")
                print(f"  {configs}")
                print(f"  {file}")
                sys.exit(1)

        print(f"Created file {new_file}")

print(f"Done!")
