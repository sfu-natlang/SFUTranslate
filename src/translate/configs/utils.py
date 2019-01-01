"""
Provides relative/absolute path information required for accessing the project resources
"""
from pathlib import Path

__author__ = "Hassan S. Shavarani"


def get_resources_dir():
    """
    :return: an object of `pathlib.PosixPath` which can be directly opened or traversed
    """
    for path in Path.cwd().parents:
        if str(path).endswith("src"):
            return Path(path.parent, "resources")
    else:
        cwd = Path.cwd()
        if str(cwd).endswith("src"):
            return Path(cwd.parent, "resources")
        else:
            raise ValueError("Unable to find /src/ directory address!")


def get_resource_file(resource_name):
    """
    :param resource_name: The name of the resource file placed in `SFUTranslate/resources` directory
    :return: an object of `pathlib.PosixPath` which can be directly opened or traversed
    """
    return Path(get_resources_dir(), resource_name)


def get_dataset_file(working_directory, resource_name, resource_extension):
    """
    :param working_directory: the directory containing the the requested resource 
    :param resource_name: the name of the resource file in the directory
    :param resource_extension: the file extension of the requested resource file. Please note that the extension param 
     must not begin with "." character as it gets already considered in the function
    :return: an object of `pathlib.PosixPath` which can be directly opened
    """
    return Path(working_directory, resource_name + "." + resource_extension)
