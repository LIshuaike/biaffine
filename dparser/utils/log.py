from dparser.utils.parallel import is_master


def log(info):
    if is_master():
        print(info)