# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from dataclasses import dataclass
from functools import reduce
from collections import namedtuple
from typing import List
import sys

from asvtorch.src.misc.singleton import Singleton

Setting = namedtuple('Setting', ['obj', 'attr', 'value'])

@dataclass
class AbstractSettings(metaclass=Singleton):

    def set_initial_settings(self, setting_file: str):
        print('Initializing settings using a settings file: {}'.format(setting_file))
        with open(setting_file) as f:
            lines = []
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if line:
                    lines.append(line)

        for line in lines:
            name, value = line.split('=', 1)                  
            name = name.strip()
            if '.' in name:
                obj, attr = name.rsplit('.', 1)
                obj = reduce(getattr, obj.split('.'), self)
            else:
                obj = self
                attr = name
            _test_existence(obj, attr)
            value = value.strip()
            ldict = {}
            exec('value = {}'.format(value), globals(), ldict)
            value = ldict['value']             
            setattr(obj, attr, value)

    def print(self):
        print(str(self))

    def _recursive_str(self, lines, prefix=''):
        variables = vars(self)
        for key in variables:
            attr = getattr(self, key)
            if isinstance(attr, AbstractSettings):
                lines.append('')
                attr._recursive_str(lines, prefix + key + '.')
            else:
                lines.append('{} = {}'.format(prefix + key, variables[key]))
        return lines

    def __str__(self):
        return '\n'.join(self._recursive_str([]))



    def load_settings(self, setting_file: str, setting_names: List[str]):
        print('Generating settings {} from file {}'.format(setting_names, setting_file))

        setting_groups = {}
        old_settings = None

        with open(setting_file) as f:
            lines = []
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if line:
                    lines.append(line)
                elif lines:
                    self._process_setting_group(setting_groups, lines)
                    lines = []
            if lines:
                self._process_setting_group(setting_groups, lines)

        for key in setting_names:
            if key not in setting_groups:
                sys.exit('ERROR: setting group named "{}" not found from the settings file {}'.format(key, setting_file))
    
        for key in setting_names:
            print('Applying settings: {}'.format(key))
            if old_settings is not None:
                self._set_settings(old_settings)

            settings = setting_groups[key]

            # Saving current config:
            old_settings = self._get_settings(settings)
            # Setting new config:
            self._set_settings(settings)
            self.post_update_call()
            string = self.get_string(settings, False)
            print(string)
            yield string

        if old_settings is not None:
            self._set_settings(old_settings)


    def _get_settings(self, settings):
        output_settings = {}
        for string, setting in settings.items():
            output_settings[string] = Setting(setting.obj, setting.attr, getattr(setting.obj, setting.attr))
        return output_settings

    def _set_settings(self, settings):
        for setting in settings.values():
            setattr(setting.obj, setting.attr, setting.value)

    def _process_setting_group(self, setting_groups, lines):

        settings = {}

        # Process name and inheritance
        parts = lines[0].split('<')
        parts = [x.strip() for x in parts]

        if parts[0] in setting_groups:
            sys.exit('Run config "{}" defined twice or more in the setting file. Remove the duplicates.'.format(parts[0]))

        for part in reversed(parts[1:]):
            if part not in setting_groups:
                sys.exit('Trying to inherit settings from "{}", which has not been defined yet'.format(part))
            settings.update(setting_groups[part])

        for line in lines[1:]:
            name, value = line.split('=', 1)
            name = name.strip()
            if '.' in name:
                obj, attr = name.rsplit('.', 1)
                obj = reduce(getattr, obj.split('.'), self)
            else:
                obj = self
                attr = name
            _test_existence(obj, attr)                
            value = value.strip()
            ldict = {}
            exec('value = {}'.format(value), globals(), ldict)
            value = ldict['value']
            settings[name] = Setting(obj, attr, value)

        setting_groups[parts[0]] = settings

    def get_string(self, settings, compact=True):
        string = ''
        delimeter = '; ' if compact else '\n'
        for s, setting in settings.items():
            string += '{} = {}{}'.format(s, setting.value, delimeter)
        if compact:
            string = string[:-1]
        return string

    # Can be overridden in child classes
    def post_update_call(self):
        pass

    # def get_value_string(self, settings):
    #     return ';'.join(str(x.value) for x in settings.values())

def _test_existence(obj, attr):
    if not hasattr(obj, attr):
        sys.exit('Setting changer error --- attribute "{}" given in config file does not exist in "{}" class!'.format(attr, obj.__class__.__name__))
