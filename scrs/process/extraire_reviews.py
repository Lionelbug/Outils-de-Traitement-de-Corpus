from browser import firefox
from bs4 import BeautifulSoup
import time
import pandas as pd

# 模拟加载更多评论
def scroll(driver, times=10, delay=2.5):
    for i in range(times):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        print(f"  → 滚动 {i + 1}/{times}")
        time.sleep(delay)

# 提取评论
def extract_reviews(driver):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    cards = soup.find_all('div', class_='apphub_Card')
    reviews = []
    for card in cards:
        try:
            tag = card.find('div', class_='title').text.strip()
            hours = card.find('div', class_='hours').text.strip()
            date = card.find('div', class_='date_posted').text.strip()
            content = card.find('div', class_='apphub_CardTextContent').text.strip()
            reviews.append({
                'Tag': tag,
                'Hours': hours,
                'Date': date,
                'Content': content
            })
        except Exception as e:
            print("    ⚠️ 跳过一条异常评论：", e)
    return reviews

# 网页爬虫
def scrape(driver, appid, scroll_times=10):
    url = f"https://steamcommunity.com/app/{appid}/reviews/?browsefilter=toprated&l=schinese"
    
    try:
        print(f"开始抓取 AppID：{appid}")
        driver.get(url)
        time.sleep(2)
        scroll(driver, times=scroll_times, delay=2.5)
        reviews = extract_reviews(driver)
        for r in reviews:
            r["AppID"] = appid  # 记录是哪一个游戏的
        return reviews
    finally:
        driver.quit()

def main():
    input_path = "data/raw/appids.csv"
    output_path = "data/raw/steam_reviews.csv"
    driver = firefox()
    df = pd.read_csv(input_path)
    all_reviews = []

    for _, row in df.iterrows():
        appid = str(row['appid'])
        try:
            reviews = scrape(driver, appid, scroll_times=10)
            all_reviews.extend(reviews)
        except Exception as e:
            print(f"⚠️ AppID {appid} 抓取失败：{e}")
    
    # 保存为csv文件
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 全部完成，共抓取 {len(df)} 条评论，保存于：{output_path}")
    else:
        print("❌ 没有抓取到任何评论")

if __name__ == '__main__':
    main()
