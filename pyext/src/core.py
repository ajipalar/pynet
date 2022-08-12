"""
Core functionality and abstractions of the pynet package used in many places

Naming Conventions for Base Classes:

    ClassNames are in CamelCase

    _setters begin with an underscore
    get_something begins with get_

This pattern allows derived classes to an attribute
and a getter and setter for that attribute.

Attribute:
    def __init__(self, attribute=None):
        self.attribute = attribute

    def get_attribute(self, *args, **kwargs):
        # get code
        ...

    def set_attribute(self, *args, **kwargs):
       # set code
       ...

    def show_attribute(self):
        # show code
        ...

DevNotes IMP Naming Conventions:
    Python 4 spaces no tabs
    Files that implement a single class should be named after that class
    free functions and macros are snake_case

    CamelCase class names
    method_name, function_name

    set_ change some stored value
    get_ create or return a value object
    create_ create a new IMP::Object
    add_, remove_, clear_ manipulate the contents of a collection of data
    show_ print things in a human readable form
    load_ and save_ or read_ and write_ move data between files
    link_ create a connection between something and an IMP::Object
    update_ change the internal state of an IMP::Object
    do_ is a virtual method as part of a non-virtual interface pattern
    handle_ take action when an event occurs
    validate_ check the state of data and print messages and throw exceptions if something is corrupted
    setup_ and teardown_ create or destroy some type invariant (e.g., the constraints on a rigid body)
    apply_ either apply a passed object to each piece of data in some collection or apply the object itself to a particular
      piece of passed data.

    
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as np
import scipy as sp
from typing import Callable, Any

from functools import partial


# TrainCase

# Function with Parameters

# Report on the Function with Paramters


class KeyValueReport:
    """
    Derived class must implemnet

    content : dict
    format_ : dict
        key_width
        gap_width
        value_width
    """

    def __init__(self):
        self.key_value_report_content = {}
        self.key_value_report_format = {
            "key_width": 10,
            "gap_width": 10,
            "value_width": 10
        }
        self.key_value_report_str = ""

    def update_key_value_report_str(self):
        self.key_value_report_str = update_key_value_report_str(
            self.key_value_report_content, **self.key_value_report_format
        )

    def clear_key_value_report_str(self):
        self.key_value_report_str = ""

    def update_key_value_report_format(self, key_width, gap_width, value_width):
        self.key_value_report_format.update({"key_width" : key_width, "gap_width" : gap_width, "value_width": value_width })


    def get_key_value_report_format(self):
        return self.key_value_report_format

    def show_key_value_report_format(self):
        print(self.key_value_report_format)

    def get_key_value_report_content(self):
        return self.key_value_report_content

    def update_key_value_report_content(self, d: dict):
        self.key_value_report_content = d

    def show_key_value_report_str(self):
        print(self.key_value_report_str)


def update_key_value_report_str(content: dict, key_width, gap_width, value_width):
    key_value_report_str = ""

    for key, val in content.items():

        assert len(str(key)) < key_width

        gap1 = key_width - len(str(key))
        line = str(key) + " " * gap1 + " " * gap_width + str(val)[0:value_width] + "\n"
        key_value_report_str += line
    return key_value_report_str

