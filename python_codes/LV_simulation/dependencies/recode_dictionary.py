import json
import sys


## function to iterate through nested dictionaries and convert unicode values to
# python strings
def recode(json_input_dict):
    """
    Recursively convert unicode strings to byte strings for Python 2 code paths.

    This handles nested dict/list/scalar structures safely, including list items
    that are plain unicode values (e.g. "save_outputs": "all").
    """
    if isinstance(json_input_dict, dict):
        for key, value in json_input_dict.items():
            json_input_dict[key] = recode(value)
        return json_input_dict

    if isinstance(json_input_dict, list):
        for idx, item in enumerate(json_input_dict):
            json_input_dict[idx] = recode(item)
        return json_input_dict

    if isinstance(json_input_dict, unicode):
        return _byteify(json_input_dict)

    return json_input_dict

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = True):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data
