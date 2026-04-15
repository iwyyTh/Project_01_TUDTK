import sys
import os
import time
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def draw_charts_from_json(json_file="benchmark_results.json"):
    """
    Hàm đọc dữ liệu từ file JSON và vẽ đồ thị mà không cần chạy lại benchmark.
    """
    # 1. Kiểm tra file tồn tại
    if not os.path.exists(json_file):
        print(f"LỖI: Không tìm thấy file '{json_file}'.")
        print("Hãy chạy benchmark.py trước để tạo dữ liệu!")
        return

    # 2. Đọc dữ liệu
    print(f"Đang đọc dữ liệu từ {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Chuyển đổi sang DataFrame và làm sạch
    df = pd.DataFrame(data)
    
    # Thay thế các chuỗi đặc biệt thành giá trị số thực để vẽ đồ thị
    df.replace("Infinity", np.inf, inplace=True)
    df.replace("NaN", np.nan, inplace=True)
    
    # Ép kiểu dữ liệu số
    cols_to_fix = ['n', 'avg_time', 'rel_error', 'condition_number']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Thiết lập giao diện vẽ
    sns.set_theme(style="whitegrid")
    
    # --- ĐỒ THỊ 1: HIỆU NĂNG THỜI GIAN ---
    plt.figure(figsize=(10, 6))
    df_random = df[df['matrix_type'] == 'random'].copy()
    if not df_random.empty:
        sns.lineplot(data=df_random, x='n', y='avg_time', hue='name', marker='o', linewidth=2)
        plt.yscale('log')
        plt.title('PHÂN TÍCH HIỆU NĂNG: Thời gian thực thi (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Kích thước ma trận (n)')
        plt.ylabel('Thời gian trung bình (s)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('performance_replot.png', dpi=300)
        print("-> Đã xuất: performance_replot.png")

    # --- ĐỒ THỊ 2: ĐỘ ỔN ĐỊNH SỐ (SAI SỐ) ---
    plt.figure(figsize=(12, 6))
    df_stability = df[df['matrix_type'].isin(['hilbert', 'spd'])].copy()
    if not df_stability.empty:
        # Tạo nhãn kết hợp để dễ phân biệt trên chú thích
        df_stability['Label'] = df_stability['name'] + " (" + df_stability['matrix_type'].str.upper() + ")"
        
        sns.lineplot(data=df_stability, x='n', y='rel_error', hue='Label', 
                     style='matrix_type', markers=True, markersize=8)
        
        plt.yscale('log')
        plt.title('PHÂN TÍCH ỔN ĐỊNH SỐ: Sai số tương đối (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Kích thước ma trận (n)')
        plt.ylabel('Sai số ||Ax-b|| / ||b||')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.savefig('stability_replot.png', dpi=300)
        print("-> Đã xuất: stability_replot.png")

    plt.show()

if __name__ == "__main__":
    print("█" * 70)
    print("Đang vẽ lại đồ thị từ dữ liệu đã lưu...")
    draw_charts_from_json("benchmark_results.json")
    print("Đã hoàn thành việc vẽ lại đồ thị.")