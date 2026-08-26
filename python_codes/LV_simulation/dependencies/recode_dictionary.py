import json
import sys
from mpi4py import MPI


## function to iterate through nested dictionaries and convert unicode values to
# python strings
def recode(json_input_dict, dictionary_path='instruction_data'):
    #print json_input_dict.values()
    for key, v in json_input_dict.items():
        value_path = '%s.%s' % (dictionary_path, key)
        
        if type(v) is dict:
            recode(v, value_path)
        else:
            # This works for a dictionary without dictionary values nested
            #for v in json_input_dict.values():

            counter = 0
            for j in v:

                if type(j) is unicode:
                    rcj = _byteify(j)
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print 'recode diagnostic path:', value_path
                        print 'type(v):', type(v)
                        print 'repr(v):', repr(v)
                        print 'counter:', counter
                        print 'type(rcj):', type(rcj)
                        print 'repr(rcj):', repr(rcj)
                    v[counter] = rcj

                counter +=1

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
