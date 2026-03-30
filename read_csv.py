import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Tạo dữ liệu từ file CSV của bạn
# (Nếu bạn có file csv, hãy dùng: df = pd.read_csv('results.csv'))
data = {
    'Combo': [
        'Facenet512_retinaface', 'Facenet512_mtcnn', 'Facenet512_opencv',
        'ArcFace_mtcnn', 'ArcFace_opencv', 'ArcFace_retinaface',
        'VGG-Face_mtcnn', 'ArcFace_mtcnn_L2', 'VGG-Face_opencv', 'VGG-Face_opencv_cos'
    ],
    'Accuracy': [69.0, 69.0, 67.5, 64.0, 61.5, 63.25, 62.25, 62.5, 61.0, 60.75],
    'FPS': [0.72, 1.30, 1.27, 0.85, 0.80, 0.36, 1.34, 1.30, 1.30, 0.90]
}

# Lấy Top 10 để vẽ cho đẹp
df = pd.DataFrame(data).head(10)

# 2. Vẽ biểu đồ
fig, ax1 = plt.subplots(figsize=(12, 6))

# Vẽ biểu đồ Cột (Accuracy)
sns.barplot(x='Combo', y='Accuracy', data=df, ax=ax1, color='#4c72b0', alpha=0.8)
ax1.set_ylabel('Độ chính xác (%)', color='#4c72b0', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#4c72b0')
ax1.set_ylim(0, 80) # Chỉnh trục Y cho thoáng

# Tạo trục Y thứ 2 để vẽ FPS
ax2 = ax1.twinx()
# Vẽ biểu đồ Đường (FPS)
sns.lineplot(x='Combo', y='FPS', data=df, ax=ax2, color='#c44e52', marker='o', linewidth=3, markersize=8)
ax2.set_ylabel('Tốc độ (FPS)', color='#c44e52', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#c44e52')
ax2.set_ylim(0, 2.0)

# Trang trí
plt.title('So sánh Độ chính xác và Tốc độ xử lý (Top 10 Model)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Cấu hình (Model + Backend)', fontsize=12)
ax1.set_xticklabels(df['Combo'], rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()