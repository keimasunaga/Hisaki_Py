import os

def get_env(pathname):
    """
    Get a path from os.environ.
    """
    return os.environ[pathname]

def set_env(pathname, path):
    """
    Set a path into os.environ.
    """
    os.environ[pathname] = path
