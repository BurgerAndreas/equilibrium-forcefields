import sys
import os

default_beginning = \
"""
# @package _global_
# ^^^ this @package directive solves any nesting problem (if this file is included in another folder)

defaults:
  # if _self_ is the first entry, compositions will overwrite this config
  # if _self_ is the last entry, this config will overwrite compositions (default)
  # https://hydra.cc/docs/1.3/upgrades/1.0_to_1.1/default_composition_order/
  - _self_
  # disable the output_subdir and logging of hydra
  - override /hydra/hydra_logging@_group_: null
  - override /hydra/job_logging@_group_: null

# disable the output_subdir and logging of hydra
hydra:
  output_subdir: null
  run:
    dir: .
"""

default_end = \
"""
# variables we can access in our code
job_name: 'results'
# job_name: ${hydra:job.name}
config_name: ${hydra:job.config_name}
# Stores the command line arguments overrides
override_dirname: ${hydra:job.override_dirname}
# Changes the current working directory to the output directory for each job
# hydra.job.chdir: False
"""

def argparse_to_hydra(
    source = "equiformer/main_qm9.py",
    target = "equiformer/config/qm9.yaml",
    write_default_beginning = True,
    write_default_end = True,
):
    # This function is used to convert argparse arguments into a hydra config file

    with open(source, "r") as f:
        lines = f.readlines()
        # get everything in the braces following parser.add_argument
        # and write to the yaml file
        with open(target, "w") as f:

            if write_default_beginning:
                f.write(default_beginning)
                # two empty lines
                f.write("\n\n")

            argument = []
            collecting_argument = False
            argument_finished = False
            for line in lines:
                if "parser.add_argument" in line:
                    collecting_argument = True
                    # add the line to the list
                    argument.append(line)
                    
                if collecting_argument:
                    if line not in argument:
                        argument.append(line)
                    if ")" in line:
                        # remove everything in quotes
                        arg_concat = "".join(argument)
                        arg_wo_quotes = ""
                        for i, s in enumerate(arg_concat.split('"')):
                            if i % 2 == 0:
                                arg_wo_quotes += s
                        # check again
                        if ")" in arg_wo_quotes:
                            argument_finished = True
                            collecting_argument = False
                
                if argument_finished:
                    # prepare the argument for writing to the yaml file
                    # print('argument:', argument)
                    argument = "".join(argument)
                    argument = argument.replace("parser.add_argument(", "")
                    argument = argument.replace("\n", "")
                    if argument[-1] == ")":
                        argument[-1] = ","
                    # argument = argument.replace(")", ",")
                    # print('  argument:', argument)

                    # get the type (if it exists)
                    if "type" in argument:
                        typehint = argument.split("type=")[1].split(",")[0]
                    if "help" in argument:
                        helphint = argument.split("help=")[1].split(",")[0]
                    else:
                        helphint = ""
                    helphint = helphint.replace('"', "")
                    # write the typehint and help to the yaml file
                    if len(helphint) > 2:
                        f.write(f"# {helphint} ({typehint})\n")
                    else:
                        f.write(f"# ({typehint})\n")

                    # get the argument name
                    arg_name = argument.split("--")[1].split(",")[0][:-1]
                    arg_name = arg_name.replace('-', "_")
                    # get the default value
                    if "default" in argument:
                        default_value = argument.split("default=")[1].split(',')[0]
                        if "None" in default_value:
                            default_value = "null"
                    else:
                        # action="store_true" means the argument is a flag
                        # and defaults to False
                        default_value = "False"
                    # write to the yaml file
                    f.write(f"{arg_name}: {default_value}\n")

                    argument = []
                    argument_finished = False
                    collecting_argument = False
            
            if write_default_end:
                # two empty lines
                f.write("\n\n")
                f.write(default_end)
        
    return


if __name__ == "__main__":
    argparse_to_hydra()
    print("Done!")