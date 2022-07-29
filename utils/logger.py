import logging
import os
import sys
import time
from logging import handlers

file_path = os.path.abspath(os.path.split(os.path.abspath(os.path.relpath(__file__)))[0] + "../../")


class Logger(object):

    def __init__(self):
        self.log_path = "{}/logs/".format(file_path)
        self.fmt = "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        self.date_fmt = "%Y-%m-%d %H:%M:%S"
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(self.fmt, datefmt=self.date_fmt)  # 设置日志格式
        self.log_filename = "{}{}.log".format(self.log_path, time.strftime("%Y-%m-%d"))
        self.logger.addHandler(self.file_handler(self.log_filename))
        self.logger.addHandler(self.console_handler())
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别

    def file_handler(self, filename):
        file_handler = logging.FileHandler(filename, encoding="utf-8")
        file_handler.setFormatter(self.formatter)
        return file_handler

    def console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler


logger = Logger().logger
