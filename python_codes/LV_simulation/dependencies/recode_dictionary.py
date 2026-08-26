import json
import sys


## Recursively convert unicode values loaded by Python 2's json module to str
# values without changing the surrounding dictionary/list/scalar structure.
def recode(json_input_dict):
    for key, value in json_input_dict.items():
        json_input_dict[key] = _recode_value(value)
    return json_input_dict


def _recode_value(value):
    if isinstance(value, dict):
        return recode(value)
    if isinstance(value, list):
        return [_recode_value(item) for item in value]
    if isinstance(value, unicode):
        return value.encode('utf-8')
    return value

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
