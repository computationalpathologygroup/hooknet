import yaml
import argparse
import sys
from pprint import pprint
import ast
import os.path

class Loader(yaml.Loader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)
        Loader.add_constructor('!include', Loader.include)
        Loader.add_constructor('!import',  Loader.include)

    def include(self, node):
        if isinstance(node, yaml.ScalarNode):
            return self.extractFile(self.construct_scalar(node))

        elif isinstance(node, yaml.SequenceNode):
            result = []
            for filename in self.construct_sequence(node):
                result += self.extractFile(filename)
            return result

        elif isinstance(node, yaml.MappingNode):
            result = {}
            for k,v in self.construct_mapping(node).iteritems():
                result[k] = self.extractFile(v)
            return result

        else:
            print("Error:: unrecognised node type in !include statement")
            raise yaml.constructor.ConstructorError

    def extractFile(self, filename):
        filepath = os.path.join(self._root, filename)
        with open(filepath, 'r') as f:
            return yaml.load(f, Loader)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    elif v.lower() in ['none', 'null']:
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse(config_file_path, notebook=False):

    config = None

    with open(config_file_path) as json_config:
        config = yaml.load(json_config, Loader=Loader)

    if notebook:
        return config

    parser = argparse.ArgumentParser(description='Hooknet')
    parser.add_argument('-c', '--config', help='config file location', required=False)

    add_arguments(parser, config)

    args = vars(parser.parse_args())

    # if custom config file is given, set config file
    if args['config']:
        with open(args['config']) as yml_config:
            config = yaml.load(yml_config, Loader=Loader)
    else:
        args['config'] = config_file_path

    set_arguments(config, args)

    print('config:')
    pprint(config)
    return config

def set_arguments(config, args, rekey=None):
    # if value is not set through arguments set value
    for key, value in config.items():
        value_type = type(value)
        setkey = ':'.join((str(rekey), str(key))) if rekey else key
        if value_type == dict:
            config[key] = set_arguments(value, args, setkey)
        if setkey in args:
            if args[setkey] is not None:
                config[key] = args[setkey]
        if value_type == str  and value.lower() == 'none':
            config[key] = None
    return config

def add_arguments(parser, config, rekey=None):
    # add keys from config file to parser arguments
    for key, value in config.items():
        value_type = type(value)
        setkey = ':'.join((str(rekey),str(key))) if rekey else key
        # list type
        if value_type == list:
            item_type = type(config[key][0]) if len(config[key]) else int
            parser.add_argument('--' + str(setkey), required=False, nargs='+', type=item_type)
        # bool type
        elif value_type == bool:
            parser.add_argument('--' + str(setkey), required=False, type=str2bool)
        # recursive add dict keys
        elif value_type == dict:
            add_arguments(parser, config[key], rekey=key)
        elif value_type == str and value == 'None':
            parser.add_argument('--' + setkey, required=False, type=str2bool)
        # int, float, string type
        else:
            parser.add_argument('--' + str(setkey), required=False, type=value_type)

