# Copyright (c) 2024 YL Feng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import json
import os
import time
from typing import Literal

import requests
from .platform_credentials import get_quafu_api_token


class Task(object):

    URL = 'https://quafu-sqc.baqis.ac.cn'

    session = requests.Session()

    def __new__(cls, *args, **kwds):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        # Task client for Quafu REST API; token comes from centralized config.
        self.token = get_quafu_api_token()
        if not self.token:
            raise ValueError('token cannot be empty!')

        self.tasks = {}

    def request(self, url: str, data: dict = {}, method: str = 'get'):
        # Thin wrapper around HTTP GET/POST with token header.
        if method == 'get':
            res = self.session.get(url, headers={'token': self.token})
        elif method == 'post':
            res = self.session.post(url,
                                    data=json.dumps(data),
                                    headers={'token': self.token})
        return json.loads(res.content.decode())

    def verify(self):
        return self.request(f'{self.URL}/task/verify')

    def query(self, tid: int = 2, chips: str = 'Baihua', status: str = 'Finished,Failed',
              start: str = '2024-04-01', end: str = time.strftime('%Y-%m-%d'),
              offset: int = 0, limit: int = 10,
              sort: Literal['taskId', 'taskName', 'chipName',
                            'status', 'submitTime'] = 'submitTime',
              order: Literal['asc', 'desc'] = 'desc'):
        return self.request(f'{self.URL}/task/query/?tid={tid}&chips={chips}&status={status}&start={start}&end={end}&offset={offset}&limit={limit}&sort={sort}&order={order}')

    def delete(self, tid: int):
        return self.request(f'{self.URL}/task/delete/{tid}')

    def result(self, tid: int, timeout: float = 0.0):
        if timeout:
            st = time.time()
            while True:
                res = self.request(f'{self.URL}/task/result/{tid}')
                if isinstance(res, dict) and res:
                    return res
                if time.time() - st > timeout:
                    raise TimeoutError(
                        f'Task {tid} result timeout after {timeout} seconds')
                time.sleep(0.2)
        else:
            time.sleep(0.2)
        return self.request(f'{self.URL}/task/result/{tid}')

    def status(self, tid: int = 0):
        time.sleep(0.2)
        return self.request(f'{self.URL}/task/status/{tid}')

    def cancel(self, tid: int):
        time.sleep(0.2)
        return self.request(f'{self.URL}/task/cancel/{tid}')

    def run(self, task: dict, repeat: int = 1):
        """run a task

        Args:
            task (dict): task description.

        Returns:
            int: task id
        """
        # Submit a circuit to the remote hardware service.
        time.sleep(0.2)
        name = task.get('name', 'MyQuantumJob')
        chip = task['chip']
        shots = task.get('shots', repeat * 1024)
        circuit = str(task['circuit'])
        tid = self.request(f'{self.URL}/task/run/?name={name}&chip={chip}&shots={shots}',
                           data={'circuit': circuit,
                                 'compile': task.get('compile', True),
                                 'options': task.get('options', {
                                     'clientip': os.getenv('CLIENT_REAL_IP', '')
                                 })},
                           method='post')
        if isinstance(tid, int):
            self.tasks[tid] = task
        return tid

