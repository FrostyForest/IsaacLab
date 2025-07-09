# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import time
import torch
import uuid
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Siglip2TextModel, Siglip2VisionModel
from transformers.image_utils import load_image
import chromadb
import open_clip
from chromadb.config import Settings
from PIL import Image
import time

# 1. 初始化 ChromaDB 客户端 (可以在内存中运行)
settings = Settings(anonymized_telemetry=False)
client = chromadb.Client(settings=settings)
# 如果你想持久化数据到磁盘，可以这样写：
# client = chromadb.PersistentClient(path="/path/to/db")

# 2. 创建一个 collection (就像创建一个 dict)
# 如果已存在，会直接获取它
collection = client.get_or_create_collection(name="robot_seen_objects")


# --- 你的主循环/函数 ---
def process_new_observation(new_embedding, new_position, env_id):
    SIMILARITY_THRESHOLD = 0.95  # 注意：ChromaDB默认用距离，需要转换或直接用距离阈值

    # 3.
    t1 = time.time()
    results = collection.query(
        query_embeddings=[new_embedding],
        n_results=1,
        include=["metadatas", "distances", "embeddings"],
        where={"env_id": env_id},
    )
    t2 = time.time()
    print("query time", t2 - t1)
    # 检查数据库是否为空或最相似的物品是否满足阈值
    is_new_item = True
    if results["ids"][0]:  # 如果查询有返回结果
        # ChromaDB的距离是L2距离，值越小越相似。我们需要转换成相似度或直接比较距离。
        # 为简单起见，我们假设距离越小越好，设定一个距离阈值
        # 这个值需要实验确定,THRESHOLD越大，不同物品被合并的可能性越小，同时物品位置信息被更新的可能性也越小，参考：red cube和yellow cube的相似度为0.82
        cosine_sim_manual_THRESHOLD = 0.87
        distance = results["distances"][0][0]
        old_embedding = results["embeddings"][0][0]
        dot_product = np.dot(new_embedding, old_embedding)
        # 2. 计算每个向量的 L2 范数 (模长)
        norm_a = np.linalg.norm(new_embedding)
        norm_b = np.linalg.norm(old_embedding)
        # 3. 根据公式计算余弦相似度
        cosine_sim_manual = dot_product / (norm_a * norm_b)
        print(cosine_sim_manual, distance)
        # 0.18,0.9

        if cosine_sim_manual > cosine_sim_manual_THRESHOLD:

            is_new_item = False
            existing_id = results["ids"][0][0]
            print(f"找到相似物品 (ID: {existing_id}), 距离: {distance:.4f}。正在更新...")

            # 4b. 覆盖原来的值
            collection.update(
                ids=[existing_id], embeddings=[new_embedding], metadatas={"position": new_position, "env_id": env_id}
            )

    if is_new_item:
        new_id = str(uuid.uuid4())
        print(f"未找到相似物品，新建索引 (ID: {new_id})")
        # 4a. 新建一个索引
        collection.add(ids=[new_id], embeddings=[new_embedding], metadatas={"position": new_position, "env_id": env_id})


def find_item_by_text(text_embedding, env_id):
    # 将文本查询编码
    # 5. 在 collection 中查询
    results = collection.query(query_embeddings=[text_embedding], n_results=1, where={"env_id": env_id})

    if results["ids"][0]:
        found_id = results["ids"][0][0]
        found_metadata = results["metadatas"][0][0]
        found_position = found_metadata.get("position")
        print(f"找到最相似的物品 (ID: {found_id}), 位置是: {found_position}")
        return found_position
    else:
        print("未在数据库中找到匹配的物品。")
        return None


clip_path = "/home/linhai/code/IsaacLab/my_code/local_siglip2/"
siglip_image_model = Siglip2VisionModel.from_pretrained(clip_path, device_map="auto").eval()
image_processor = AutoProcessor.from_pretrained(clip_path, device_map="auto")
siglip_text_model = Siglip2TextModel.from_pretrained(clip_path, device_map="auto").eval()
text_processor = AutoTokenizer.from_pretrained(clip_path, device_map="auto")

image = image_processor(
    images=load_image("my_code/segment_anything/output_objects/object_3.png"), max_num_patches=64, return_tensors="pt"
).to("cuda")
# tokenizer = open_clip.get_tokenizer("TULIP-B-16-224")
text = text_processor(
    text=["a yellow cube", "yellow_cube", "a red cube"], padding="max_length", max_length=64, return_tensors="pt"
).to("cuda")
image2 = image_processor(
    images=load_image("my_code/segment_anything/output_objects/object_2.png"), max_num_patches=64, return_tensors="pt"
).to("cuda")
image3 = image_processor(
    images=load_image("my_code/segment_anything/output_objects/object_8.png"), max_num_patches=64, return_tensors="pt"
).to("cuda")

with torch.no_grad(), torch.autocast("cuda"):
    image_features = siglip_image_model(**image).pooler_output.squeeze(0).cpu().numpy()
    text_features = siglip_text_model(**text).pooler_output.cpu().numpy()
    image_features2 = siglip_image_model(**image2).pooler_output.squeeze(0).cpu().numpy()
    image_features3 = siglip_image_model(**image3).pooler_output.squeeze(0).cpu().numpy()
    f0 = text_features[0]
    f1 = text_features[1]
    f2 = text_features[2]


process_new_observation(image_features2, 0, 0)
# process_new_observation(image_features3, 1,0)
# process_new_observation(image_features3,2)
process_new_observation(image_features, 2, 0)
find_item_by_text(f0, 0)
