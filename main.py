#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# July 2020

from utils import logging_time

import time
@logging_time
def hello(arg):
    time.sleep(1)

    print("[!] hello {}".format(arg))

for k in range(10):
    print(hello(k))