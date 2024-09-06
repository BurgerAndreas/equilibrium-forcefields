import os
import shutil
import pathlib

def rename_folders(root_dir, dryrun=True):
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == 'aspirin':
                # don't rename the directory "aspirin", 
                # only directories containing "aspirin"
                print("Skipping folder 'aspirin':", os.path.join(dirpath, dirname))
                continue
            if 'aspirin' in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace('aspirin', '')
                new_path = os.path.join(dirpath, new_dirname)
                
                # Rename the directory
                if not dryrun:
                    # 1
                    # os.rename(old_path, new_path)
                    # 2
                    shutil.move(old_path, new_path)
                    # 3
                    # path = pathlib.Path(old_path)
                    # path.rename(new_path)
                print(f"Renamed folder {dirname}:\n {old_path} ->\n {new_path}")

if __name__ == "__main__":

    dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
    # Set your root directory path here
    root_directory = os.path.join(
        dir_of_this_script, "..", "models/md17"
    )
    root_directory = os.path.abspath(root_directory)
    
    print(f"Renaming folders in {root_directory}")
    rename_folders(root_directory, dryrun=False)
