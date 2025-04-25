import shutil
from selenium import webdriver
from selenium.webdriver.firefox.service import Service

def firefox():
    # 检测 geckodriver
    driver_path = shutil.which("geckodriver")
    if not driver_path:
        raise RuntimeError("❌ geckodriver not found in PATH. Please install it and add to PATH.")
    print(f"✅ geckodriver found at {driver_path}")

    # 可选：检测 Firefox
    browser_path = shutil.which("firefox")
    if not browser_path:
        raise RuntimeError("❌ Firefox not found. Please install Firefox.")
    print(f"✅ Firefox found at {browser_path}")

    # 设置 Firefox 启动参数
    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')  # 如果你不想打开图形窗口

    # 启动 Firefox
    service = Service(driver_path)
    driver = webdriver.Firefox(service=service, options=options)
    return driver