from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from bson import ObjectId

# Khởi tạo app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://animetangobackend.onrender.com"],  # Cho phép tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối MongoDB
client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_favorites = pd.DataFrame(list(db["UserFavorites"].find()))  # Thay đổi thành UserFavorites
df_anime = pd.DataFrame(list(db["Anime"].find()))

user_anime_matrix = pd.DataFrame(
    [
        (user["User_id"], anime_id, 1)  # Đánh giá anime = 1 nếu người dùng yêu thích
        for user in df_favorites.to_dict(orient="records")
        for anime_id in user["favorites"]
    ],
    columns=["User_id", "Anime_id", "Rating"]
)

# Pivot bảng để tạo ma trận sparse
animes_users = user_anime_matrix.pivot(index="User_id", columns="Anime_id", values="Rating").fillna(0)
mat_anime = csr_matrix(animes_users.values)

# Bước 2: Huấn luyện mô hình KNN để tìm các người dùng tương tự
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)  # Bạn có thể thay đổi số lượng người dùng gần nhất (n_neighbors)
model.fit(mat_anime)

from bson import ObjectId

def jsonable(data):
    if isinstance(data, list):
        return [jsonable(item) for item in data]
    elif isinstance(data, dict):
        return {key: jsonable(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data
# API POST để gợi ý anime
@app.post("/")
async def recommend(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    # Tìm người dùng tương tự
    user_idx = animes_users.index.get_loc(user_id)
    distances, indices = model.kneighbors(mat_anime[user_idx], n_neighbors=n)

    recommended_animes = set()
    for i in indices.flatten():
        if i != user_idx:  # Loại bỏ chính người dùng
            similar_user = animes_users.iloc[i]
            # Tìm anime mà người dùng này đã xem nhưng người dùng ban đầu chưa xem
            for anime_id, rating in similar_user.items():
                if rating > 0 and anime_id not in animes_users.iloc[user_idx][animes_users.columns != anime_id]:
                    recommended_animes.add(anime_id)

    # Lấy thông tin anime từ df_anime
    recommendations = []
    for anime_id in recommended_animes:
        anime_data = df_anime[df_anime['Anime_id'] == anime_id].iloc[0].to_dict()
        recommendations.append(anime_data)

    return jsonable(recommendations)

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sẽ cung cấp cổng trong biến PORT
    uvicorn.run("knn2:app", host="0.0.0.0", port=port)
