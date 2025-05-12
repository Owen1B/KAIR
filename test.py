from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from tqdm import tqdm
import time

# 配置基本logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 使用tqdm的logging重定向
with logging_redirect_tqdm():
    for i in tqdm(range(10)):
        time.sleep(0.5)
        if i % 3 == 0:
            logger.info(f"处理到了第 {i} 项")