import pandas as pd
import re

# 读取原始数据
df = pd.read_csv('../data/raw/steam_reviews.csv')

def clean_comment(text):
    # 去除“发布于：xxxx 年 xx 月 xx 日”以及紧接着的一行“抢先体验版本评测”（如果有）
    text = re.sub(r'发布于：\d{4} 年 \d+ 月 \d+ 日\s*抢先体验版本评测\s*', '', text)
    text = re.sub(r'发布于：\d{4} 年 \d+ 月 \d+ 日', '', text)  # 如果没有“抢先体验版本评测”
    return text.strip()

# 清洗评论字段
df['Comment'] = df['Comment'].apply(clean_comment)

# 保存新文件
df.to_csv('../data/clean/steam_reviews_cleaned.csv', index=False, encoding='utf-8-sig')
print("清洗完成，已保存为 steam_reviews_cleaned.csv")