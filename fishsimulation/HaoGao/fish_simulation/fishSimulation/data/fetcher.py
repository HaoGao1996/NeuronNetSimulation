def get_data_rd(dir=None):
    """
    Get data relative directory.
    :param dir:
        relative directory for data example
    :return:
    """
    from os.path import join, dirname
    DATA_DIR = join(dirname(__file__), 'files')

    if dir is None:
        return DATA_DIR
    else:
        data_rd = join(DATA_DIR, dir)
        return DATA_DIR, data_rd