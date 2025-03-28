import shutil
import os

def delete_folders(folders):
    """
    Recursively deletes multiple folders.

    :param folders: List of folder paths to delete.
    """
    for folder in folders:
        if os.path.exists(folder) and os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                print(f"Deleted: {folder}")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")
        else:
            print(f"Skipping (not found): {folder}")

# Example usage
folders_to_delete = [
    "/workspace/split-dataset",
    "/workspace/dataset",
    "/workspace/output"
]

delete_folders(folders_to_delete)