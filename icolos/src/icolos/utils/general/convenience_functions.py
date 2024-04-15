from icolos.utils.general.files_paths import attach_root_path


# dictionary convenience functions
# ---------


def nested_get(dictionary: dict, keys: list, default=None):
    # recursively walk through nested dict, pull the value corresponding to the specified key(s)
    if not isinstance(keys, list):
        keys = [keys]
    if dictionary is None:
        return default
    if not keys:
        return dictionary
    return nested_get(dictionary.get(keys[0]), keys[1:], default)


def in_keys(dictionary: dict, keys: list) -> bool:
    if not isinstance(keys, list):
        keys = [keys]

    _dict = dictionary
    for key in keys:
        try:
            _dict = _dict[key]
        except KeyError:
            return False
    return True


# parsing "setup.py"
# ---------


def parse_setuppy():
    path = attach_root_path("setup.py")
    parsed_dict = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "name" in line:
                parsed_dict["name"] = line[line.find('"') + len('"') : line.rfind('"')]
            if "version" in line:
                parsed_dict["version"] = line[
                    line.find('"') + len('"') : line.rfind('"')
                ]
            if "license" in line:
                parsed_dict["license"] = line[
                    line.find('"') + len('"') : line.rfind('"')
                ]
            if "author" in line:
                parsed_dict["author"] = line[
                    line.find('"') + len('"') : line.rfind('"')
                ]
    return parsed_dict


def ensure_list(inp) -> list:
    return inp if isinstance(inp, list) else [inp]
