from huggingface_hub import create_repo, upload_folder

repo_id = "kurakurai/Luth-0.6B-Scholar-0.5-v2"
folder_path = "merged-output"

# Create the repo (does nothing if it already exists and `exist_ok=True`)
create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# Upload the folder
upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="model")
