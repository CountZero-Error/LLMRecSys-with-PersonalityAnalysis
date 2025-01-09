from flask import Flask, request, jsonify
import json
import numpy as np
import sys
import os
from datasets import load_dataset

# read ssgb
# 获取当前文件（backend.py）的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取 Model_training 文件夹的路径
model_training_path = os.path.join(current_dir, '..', 'Model_training')

# 添加 Model_training 路径到 sys.path
if model_training_path not in sys.path:
    sys.path.append(model_training_path)

from SSGB import preprocessing, GraphClustering


# 初始化 Flask 应用
app = Flask(__name__)

fi = '/Users/cookie/Desktop/FYP/1000k/parameter_labeled.csv'

access_token = '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'
original_embedding = load_dataset(
    "CookieLyu/Category_Codes",
    revision="1000k_average_embedded",
    token=access_token
)

augmented_embedding = load_dataset(
    "CookieLyu/Category_Codes",
    revision="1000k_average_embedded_aug",
    token=access_token
)

label_types = ['Night_owl', 'Early_bird', 'Decisive', 'Brand_loyalty', 'Maker', 'Homebody', 'Culinarian', 'Geek',
               'Photophile', 'Media_Aficionado', 'Audiophile', 'Fashionista', 'Lifestyle', 'Car_Enthusiast',
               'Caregiver', 'Health_Enthusiast', 'Farm', 'Sport', 'high_consumer', 'Mid_Consumer']

# 配置和加载模型
preprocessor = preprocessing(fi=fi, label_types=label_types, original_embedding=original_embedding, augmented_embedding=augmented_embedding)
OneHotLabels = preprocessor.get_OneHotEncoder_label()
all_features, seed_indices, merged_labels, consistency_loss = preprocessor.run()

clustering_model = GraphClustering(features=all_features, k_neighbors=5, threshold=0.1,
                                    labels=merged_labels, seed_indices=seed_indices, consistency_loss=consistency_loss,
                                    step=3, OneHotLabels=OneHotLabels)

@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.get_json()

    try:
        # Step 1: 预处理单个用户
        user_features = preprocessor.preprocess_single_user(user_data)

        # Step 2: 使用模型预测
        predicted_labels = clustering_model.predict_single_user(user_features)

        return jsonify({"labels": predicted_labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
