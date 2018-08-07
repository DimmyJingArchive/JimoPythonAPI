import importlib
import glob
import time
import json
import sys
import re
import os
sys.path.insert(0, './Modules')

init_list = False
module_timestamp = {}
module_list = {}
functions = {}


def update_list():
    global init_list
    try:
        if not init_list:
            init_list = True
            data = {}
        else:
            data = json.load(open('Modules/.modules.json'))
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data = json.loads('{}')
    modules = sum([glob.glob('Modules/' + i) for i in ['*.py', '*.pyc', '*.pyo']], [])
    modules = [os.path.basename(f) for f in modules if os.path.isfile(f)]

    for module in modules:
        split_name = os.path.splitext(module)[0]
        try:
            if (data[split_name]['timestamp'] >
                    os.path.getmtime('./Modules/' + module)):
                continue
        except KeyError:
            pass
        new_module = importlib.import_module(split_name)
        try:
            module_name = getattr(new_module, 'module_name')
        except AttributeError:
            continue
        data[split_name] = {}
        data[split_name]['timestamp'] = os.path.getmtime('./Modules/' + module)
        data[split_name]['module_name'] = module_name

    json.dump(data, open('Modules/.modules.json', 'w'))

    param = re.compile(r'{([^}]*)}')
    for module in data:
        for module_name in data[module]['module_name']:
            module_list[param.sub('(\\w+)', module_name[0])] = {
                                'file_name': module,
                                'func_name': module_name[1],
                                'param_name': param.findall(module_name[0]),
                                'timestamp': data[module]['timestamp']}


def update_all_modules():
    global functions
    functions = {}
    update_list()
    for i in module_list:
        update_command(module_name=i)


def update_command(command_name):
    new_module = importlib.import_module(module_list[command_name]['file_name'])
    try:
        if (os.path.getmtime(glob.glob('Modules/' + module_list[command_name]['file_name'] + '.*')[0]) >
                module_timestamp[module_list[command_name]['file_name']]):
            new_module = importlib.reload(new_module)
        module_timestamp[command_name] = time.time()
    except (KeyError, IndexError):
        module_timestamp[command_name] = time.time()
    func = getattr(new_module, module_list[command_name]['func_name'], None)
    if func is not None:
        functions[command_name] = func
    else:
        print("File '{0}' has no function '{0}'.".format(module_list[command_name]['file_name'], module_list[command_name]['func_name']))


def execute(exec_module_name, *args, **kwargs):
    if (os.path.getmtime(glob.glob('Modules/'+module_list[exec_module_name]['file_name']+'.*')[0]) >
            module_list[exec_module_name]['timestamp']):
        update_list()
        update_command(exec_module_name)
    try:
        functions[exec_module_name]
    except KeyError:
        update_list()
        update_command(exec_module_name)
        try:
            functions[exec_module_name]
        except KeyError:
            return ('There are no modules that contain the function: {}'.format(exec_module_name))
    return functions[exec_module_name](*args, **kwargs)


update_list()
