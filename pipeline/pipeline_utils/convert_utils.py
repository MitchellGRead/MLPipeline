def dict_to_list(data: dict, keys: list[str]) -> list[dict[str, any]]:
    """Convert a dictionary to a list of dictionaries.

    Args:
        data (dict): input dictionary.
        keys (list[str]): keys to include in the output list of dictionaries.

    Returns:
        list[dict[str, any]]: output list of dictionaries.
    """
    list_of_dicts = []
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts
