def getattr_or_default(obj, attr, default=None):
    value = getattr(obj, attr, default)
    return value if value is not None else default