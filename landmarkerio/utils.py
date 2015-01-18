import os.path as p


def parse_username_and_password_file(path):
    with open(p.abspath(p.expanduser(path)), 'rb') as f:
        up = f.readlines()
    return tuple([l.strip() for l in up][:2])
