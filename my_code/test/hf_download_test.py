from huggingface_hub import snapshot_download

repo_id = "google/siglip2-base-patch16-naflex"
local_folder = "./my_local_siglip_model" # 你希望保存文件的本地文件夹路径

proxy={
    'http':"http://127.0.0.1:7890",
    'https':"http://127.0.0.1:7890",
}
snapshot_download(
    repo_id=repo_id,
    local_dir=local_folder,
    local_dir_use_symlinks=False, # 或者 True，如果你想用符号链接
    # token="YOUR_HF_TOKEN" # 如果模型是私有的或需要认证
    proxies=proxy
)
print(f"Files downloaded to {local_folder}")