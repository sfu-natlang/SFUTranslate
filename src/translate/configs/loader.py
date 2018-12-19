from yaml import load


def find(yaml_element, searching_key):
    """
    A generator function that searches the whole yaml element and returns all of the values for the existing key
    """
    if searching_key in yaml_element:
        yield yaml_element[searching_key]
    for k, v in yaml_element.items():
        if isinstance(v, dict):
            for i in find(v, searching_key):
                yield i


class ConfigLoader:
    def __init__(self, config_file):
        """
        The class to receive the address of the config file and retrieve the values in it.
        You can access the values of configuration "a.b.c" in yaml file simply by calling ConfigReader_instance["a.b.c"]
        :param config_file: an instance of `pathlib.Path`, if you only know the file address (e.g. /path/to/config.yaml)
                you can convert the address to a Path instance using Path('/path/to/config.yaml')
        """
        self._config_data = load(config_file.open().read())

    def get(self, key, default_value=None, must_exist=False):
        """
        :param key: a dot-separated key identifier representing the nested parent keys in the config file
                (the key must be a string)
        :param default_value: the value to be returned if the requested key does not exist
        :param must_exist: if True ignores the default_value and if cannot find the key raises Exception
        :return: the value of the requested key, WARNING: always the first occurrence of the key will get returned
        """
        assert type(key) == str
        if not key or key.isspace():
            return default_value
        if must_exist:
            default_value = None
        nested_keys = key.split(".")
        current_node = self._config_data
        for n_key in nested_keys:
            current_node = next(find(current_node, n_key), default_value)
            if current_node == default_value:
                break
        if must_exist and current_node is None:
            raise ValueError("The configuration value {} must exist in your configuration file!".format(key))
        return current_node
