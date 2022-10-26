import os
import json
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

def get_absolute_path(path, base=ROOT):
    """Return absolute path
        Args:
            path (str): a given path
            base (str): prefix
        Returns:
            path (str) abs(base + path)
    """
    return os.path.abspath(os.path.join(base, path))

def get_config(test=False):
    """Get configs"""
    if test:
        env = 'test'
    else:
        env_var = os.getenv("ENV_VAR")
        if env_var is None:
            env = 'local'
        else:
            env = env_var.split('-')[-1]

    fname = get_absolute_path(f'configs/config-{env}.json')
    with open(fname) as f:
        return json.load(f)

def get_logger(name=None):
    """Set Steam Logger"""
    if name is None:
        name = __name__
    logger = logging.getLogger(name=name)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter('%(asctime)s : %(processName)s : %(levelname)s : %(message)s'))
    logger.addHandler(log_handler)
    return logger

def set_logger_level(logger, level):
    """Set logger level
        Args:
            logger (logging.RootLogger): a logger
            level (str): DEBUG, INFO, WARNING, ERROR
    """
    try:
        logger.setLevel(logging.getLevelName(level))
    except:
        logger.warning(f"Failed to set logging level to {level}")

def get_mb(byte):
    """Turn bytes into MBs"""
    return int(byte/(2**20))
