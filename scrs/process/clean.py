import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 去除 Content 中可能重复出现的日期字段
    df['Content'] = df['Content'].str.replace(
        r'发布于[:：]?\s*(\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*日',
        '', 
        regex=True
    )
    df['Content'] = df['Content'].str.replace(r'抢先体验版本评测', '', regex=False)
    df['Content'] = df['Content'].str.replace(r'免费获取的产品', '', regex=False)
    
    # 去除换行符、多余空格
    df['Content'] = df['Content'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 清洗 Date 字段：只保留纯日期
    df['Date'] = df['Date'].str.replace(r'发布于：', '', regex=False).str.strip()
    
    # 清洗 Hours 字段：去掉“总时数”
    df['Hours'] = df['Hours'].str.replace(r'总时数', '', regex=False).str.strip()
    
    # 只保留包含中文的评论
    df = df[df['Content'].str.contains(r'[\u4e00-\u9fa5]', regex=True)]
    
    return df

df = pd.read_csv("../../data/raw/steam_reviews.csv")
df = clean(df)
df.to_csv('../../data/clean/steam_reviews.csv', index=False, encoding='utf-8-sig')
print("清洗完成，已保存为 steam_reviews_cleaned.csv")