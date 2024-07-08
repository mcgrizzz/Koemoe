from __future__ import annotations

import os

import sys
import warnings
import enlighten
import contextlib
import pkg_resources

from pathlib import Path
from enlighten._counter import Counter 
from functools import partial
from dataclasses import dataclass, field

progress_colors = [[237, 226, 225], [237, 205, 202], [237, 172, 166], [238, 124, 114]]

class TieredCounter(Counter):
    level: int
    prefix: str
    parent: str
    trimmed_desc: str
    keep_children: bool
    children: list[TieredCounter]
    
    __slots__ = ('all_fields', 'level', 'prefix', 'parent', 'trimmed_desc', 'keep_children', 'children', 'bar_format', 'counter_format', 'desc', 'fields',
                 'offset', 'series', 'total', 'unit', '_fields', '_subcounters')
    
    _repr_attrs = ('level', 'parent', 'trimmed_desc', 'count', 'total', 'unit', 'color')
    
    def __init__(self, **kwargs):
        self.level = kwargs.pop("level", 0)
        self.prefix = kwargs.pop("prefix", "") * self.level
        self.keep_children = kwargs.pop("keep_children", True)
        self.parent = kwargs.pop("parent", "")
        self.children = []
        
        desc = kwargs.pop("desc", "")
        self.trimmed_desc = desc
        
        desc = self.prefix + desc
        kwargs["desc"] = desc
        super().__init__(**kwargs)
    
    def update(self, incr=1, force=False, **fields):
        prog_fract = (self.count + incr)/self.total
        index = int((len(progress_colors) - 1)*prog_fract)
        self.color = progress_colors[index]
        super().update(incr, force, **fields)
        
    def add_child(self, child: TieredCounter):
        self.children.append(child)

    def close(self, clear=False):
        #print(f'Closing {self.to_string()}')
        super().close(clear)
        self.close_children()
                
    def close_children(self):
        for child in self.children:
            #print(f'Closing Child {child.to_string()}')
            if not child._closed:
                child.leave=self.keep_children
                child.close(clear=(not self.keep_children))
    
    def to_string(self):
        return f'({self.level}, [{self.parent}/{self.trimmed_desc}])'
        
class TieredManager(enlighten.Manager): 
    
    def __init__(self, **kwargs):
        self.counter_list = []
        self.prefix = kwargs.pop("prefix", "    ")
        super().__init__(**kwargs)
    
    def counter(self, level=0, keep_children=True, position=None, **kwargs) -> TieredCounter:
        counter_list = self.counter_list
        #find parent, AKA the first counter above it in the list that has a lower level
        parent: TieredCounter = None
        if len(counter_list) > 0:
            for i in reversed(range(0, len(counter_list))):
                tiered: TieredCounter = counter_list[i]
                if tiered.level < level:
                    parent = tiered
                    break
            
        kwargs["level"] = level
        kwargs["keep_children"] = keep_children
        kwargs["prefix"] = self.prefix
        
        parent_name = parent.trimmed_desc if parent else ""
        kwargs["parent"] = parent_name
        
        counter: TieredCounter = self._add_counter(self.counter_class, position=position, **kwargs)
        self.counter_list.append(counter)
        
        if parent:
            if parent.keep_children:
                kwargs["leave"] = True
            else:
                kwargs["leave"] = False
            parent.add_child(counter)
            
        return counter

def get_manager(stream=None, **kwargs):
    stream = sys.__stdout__ if stream is None else stream
    isatty = hasattr(stream, 'isatty') and stream.isatty()
    kwargs['enabled'] = isatty and kwargs.get('enabled', True)
    return TieredManager(stream=stream, counter_class=TieredCounter, **kwargs)

manager: TieredManager = get_manager()

@contextlib.contextmanager
def nowarning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
        
def get_latest_model():
    models = os.listdir(Path("./model"))
    models.remove('.gitkeep')
    latest_version = ("None", pkg_resources.parse_version("0.0"))
    for file_name in models:
        file = Path(file_name).stem
        version = pkg_resources.parse_version(file.split("_")[-1])
        if version > latest_version[1]:
            latest_version = (file_name, version)
    return latest_version[0]