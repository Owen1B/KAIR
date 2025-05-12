import sys
import datetime
import logging
from tqdm import tqdm  # 改为仅导入 tqdm


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


# 自定义与 tqdm 兼容的日志处理程序
class TqdmCompatibleStreamHandler(logging.StreamHandler):
    """
    使用 tqdm.write 替代 StreamHandler.emit，使日志与 tqdm 进度条兼容
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


'''
# --------------------------------------------
# logger
# --------------------------------------------
'''


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist! Current TQDM integration might not apply if pre-configured differently.')
    else:
        print('LogHandlers setup with TQDM integration!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)

        # 使用自定义的 tqdm 兼容处理程序替代标准流处理程序
        tqdm_sh = TqdmCompatibleStreamHandler()
        tqdm_sh.setFormatter(formatter)
        log.addHandler(tqdm_sh)


'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass
