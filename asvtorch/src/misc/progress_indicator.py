# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import random
import time
import os

class ProgressIndicator():
    def __init__(self, input_list, n_updates):
        if os.path.exists("temp_progress.txt"):
            os.remove("temp_progress.txt")
        self.n_updates = n_updates
        self.update_inputs = set(random.sample(input_list, n_updates))
        self.start_time = time.time()

    def update(self, step_input):
        if step_input in self.update_inputs:
            with open('temp_progress.txt', 'a+') as f:
                f.write('|')
                f.seek(0)
                line = f.readline().strip()
                percentage = len(line) / self.n_updates * 100
                elapsed = time.time() - self.start_time
                left = elapsed / percentage * 100 - elapsed
                line = '<{:{width}}>'.format(line, width=self.n_updates)
                print('{} {:5.1f} %  Elapsed time = {:.0f} s, Time left = {:.0f} s'.format(line, percentage, elapsed, left))
