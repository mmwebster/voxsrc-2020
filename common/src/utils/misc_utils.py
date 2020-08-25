#! /usr/bin/python
# -*- encoding: utf-8 -*-
import time

# @brief Simple class to throttle print statements that might spam. Necessary for
#        cluster runs, since spammed logs will prevent any from rendering on
#        kubeflow GUI or sent to wandb
# @note Will only print if invoked, a trailing call that doesn't meet the period
#       threshold will never be printed
class print_throttler():
    def __init__(self, min_print_period_secs=10):
        self.min_print_period_secs = min_print_period_secs
        self.last_call = 0

    def throttle_print(self, print_str):
        if (time.time() - self.last_call) >= self.min_print_period_secs:
            print(print_str)
            self.last_call = time.time()
