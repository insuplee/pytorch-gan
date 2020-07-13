#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# July 2020

import time

def logging_time(original_func):
    def wrapper_func(*args, **kwargs):
        start_time = time.time()
        print("[!] START {}".format(original_func))
        res = original_func(*args, **kwargs)
        print(">> time: {:.2f} seconds".format(time.time()-start_time))
        return res
    return wrapper_func