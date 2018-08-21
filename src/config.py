import argparse
import json

import collections

from jinja2 import Environment, FileSystemLoader, StrictUndefined

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            if '+' in v:
                v = [float(x) for x in v.split('+')]
            try:
                d[k] = type(d[k])(v)
            except (TypeError, ValueError) as e:
                raise TypeError(e) # types not compatible
            except KeyError as e:
                d[k] = v # No matching key in dict
    return d


class ConfigParse(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options_dict = {}
        for overrides in values.split(','):
            k, v = overrides.split('=')
            k_parts = k.split('.')
            dic = options_dict
            for key in k_parts[:-1]:
                dic = dic.setdefault(key, {})
            dic[k_parts[-1]] = v
        setattr(namespace, self.dest, options_dict)


def get_config_overrides():
    parser = argparse.ArgumentParser(description='Experiments for aggregated momentum')
    parser.add_argument('config', help='Base config file')
    parser.add_argument('-o', action=ConfigParse, help='Config option overrides. Comma separated, e.g. optim.lr_init=1.0,optim.lr_decay=0.1')
    args, template_args = parser.parse_known_args()
    template_dict = dict(zip(template_args[:-1:2], template_args[1::2]))
    template_dict = { k.lstrip('-'): v for k,v in template_dict.items() }
    return args,template_dict

def process_config(verbose=True):
    args, template_args = get_config_overrides()

    with open(args.config, 'r') as f:
        template = f.read()

    env = Environment(loader=FileSystemLoader('configs/templates/'),
                      undefined=StrictUndefined)

    config = json.loads(env.from_string(template).render(**template_args))

    if args.o is not None:
        print(args.o)
        config = update(config, args.o)

    if verbose:
        import pprint
        pp = pprint.PrettyPrinter()
        print('-------- Config --------')
        pp.pprint(config)
        print('------------------------')
    return config
