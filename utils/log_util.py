import logging


def init_file_logger(file='./logs/log.txt', name='file-logger', level=logging.DEBUG) -> logging.Logger:
    """
    初始化文件日志记录器
    :param file: 日志文件
    :param name: 日志记录器名称
    :param level: 日志级别
    :return: 日志记录器（单例、全局唯一）
    """
    # 日志记录器
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    # 文件处理器
    file_handler = logging.FileHandler(file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler_formatter = logging.Formatter('[%(asctime)s] - [%(levelname)7s] - [%(module)s]: %(message)s')
    file_handler.setFormatter(file_handler_formatter)
    logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    logger = init_file_logger(name='test')
    logger.info('test1')
    logger.warning('test2')
    logger.error('test3')