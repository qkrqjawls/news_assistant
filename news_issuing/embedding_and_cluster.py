# embedding_and_cluster.py

import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from utils import load_all_json_from_folder, preprocess_text

def compute_embeddings(items, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """SBERT 모델로 각 기사(제목+본문)의 임베딩 계산"""
    model = SentenceTransformer(model_name)
    texts = [preprocess_text(x.get("title",""), x.get("body","")) for x in items]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

def cluster_and_save(input_folder: str, output_path: str, dist_thresh=0.6):
    """newsdata/ → 임베딩 → AgglomerativeClustering → clustered.json 저장"""
    items = load_all_json_from_folder(input_folder)
    embeddings = compute_embeddings(items)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="cosine",
        linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(embeddings)

    # 결과에 cluster_id를 붙여 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = []
    for item, lbl in zip(items, labels):
        item["cluster_id"] = int(lbl)
        out.append(item)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"▶ clustered saved to {output_path} (threshold={dist_thresh})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_folder", default="newsdata",
                   help="원본 JSON들이 담긴 폴더")
    p.add_argument("--output_path", default="clustered.json",
                   help="클러스터링 결과 JSON 경로")
    p.add_argument("--dist_thresh", type=float, default=0.6,
                   help="Agglomerative distance_threshold")
    args = p.parse_args()
    cluster_and_save(args.input_folder, args.output_path, args.dist_thresh)
