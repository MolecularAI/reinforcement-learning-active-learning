def dataclass_from_dict(cls, obj):
    """Constructs an instance of a dataclass from a given dict object.

    :param cls: dataclass to instantiate
    :param obj: dictionary to use as input data
    :return: dataclass instance
    """

    # This implementation requires no external dependencies. Adapted from:
    #   - https://stackoverflow.com/a/54769644
    #   - https://gist.github.com/gatopeich/1efd3e1e4269e1e98fae9983bb914f22
    # TODO(alex): discuss and decide on a proper parsing and validation library.
    # Possible candidates include: apischema, pydantic v2.

    try:
        fieldtypes = cls.__annotations__
        return cls(**{f: dataclass_from_dict(fieldtypes[f], obj[f]) for f in obj})
    except AttributeError:
        if isinstance(obj, (tuple, list)):
            return [dataclass_from_dict(cls.__args__[0], f) for f in obj]
        return obj
