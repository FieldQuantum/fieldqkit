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


class Task(object):

    URL = 'https://quafu-sqc.baqis.ac.cn'

    session = requests.Session()

    def __new__(cls, *args, **kwds):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        # Task client for Quafu REST API; token can come from env or argument.
        token = "dJCvzfuPQ{dsDOz1ysbZSfMY3{NSB9U3usgj:BkMo:`/:OkNyl{N2ZkM5BEO3F{O{d{OypkJxiY[jxjJ5BkPxBkPyFEJ2FUMzBUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"
        self.token = os.getenv('QPU_API_TOKEN', token)
        assert self.token, 'token cannot be empty!'

        v = self.verify()
        if isinstance(v, dict):
            raise Exception(f'{v}')

        self.tasks = {}
        self.cache = {}

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

    def backend(self, chip: str, show_couplers_fidelity: bool = False, show_quibts_attributes: Literal['T1', 'T2', 'fidelity', 'frequancy', ''] = '', highlight_nodes: list = [], save_svg_fname: str | None = None):
        try:
            from .backend import Backend
            bk = Backend(chip)
            bk.draw(show_couplers_fidelity,
                    show_quibts_attributes,
                    highlight_nodes,
                    save_svg_fname)
            return bk.chip_info
        except Exception as e:
            print(f'{e}, install it using "pip install quarkcircuit"')

        # try:
        #     self.cache[chip]
        # except KeyError as e:
        #     self.cache[chip] = info = bk.chip_info

    def _backend(self, chip: str, draw: str = '', refresh: bool = False):
        if refresh:
            self.cache.clear()
        try:
            info = self.cache[chip]
        except KeyError as e:
            info = self.request(f'{self.URL}/task/backend/{chip}')
            self.cache[chip] = info

        return self.__plot(info, draw)

    def __plot(self, info: dict, draw: str):
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure(figsize=[14, 11])
        graph = nx.Graph()
        nodepos, nodecolor, nodelabel, edgecolor, edgelabel = {}, {}, {}, {}, {}
        mapping = info['mapping']
        sqi, tqi = info['single_qubit_info'], info['two_qubit_info']
        for i, edge in enumerate(tqi.keys()):
            qs, qe = edge.split('_')
            for q in [qs, qe]:
                nodepos[q] = [int(int(q[3:])), 24 - int(q[1:3]) % 24]
                nodecolor[q] = sqi[q].get(draw, 0) if draw else 'lightblue'
                nodelabel[q] = sqi[q].get(draw, '') if draw else mapping[q]
                if q in info['unavailable']:
                    nodelabel[q] = ''

            graph.add_edge(qs, qe)
            edgecolor[(qs, qe)] = 'blue'  # tqi[edge]['CZ']['fidelity']
            edgelabel[(qs, qe)] = tqi[edge]['CZ']['fidelity']

        nx.draw(graph,
                pos=nodepos,
                with_labels=True,
                labels=nodelabel,
                font_size=8,
                font_color='k',
                node_size=600,
                edgecolors='blue',  # for node edge
                node_color=[nodecolor[q] for q in nodecolor.keys()],

                edge_color=[edgecolor[q] for q in edgecolor.keys()],
                width=9.0,
                # alpha=1,
                cmap=plt.cm.Wistia
                )

        texts = nx.draw_networkx_edge_labels(graph,
                                             pos=nodepos,
                                             edge_labels=edgelabel,
                                             bbox=dict(boxstyle='round',
                                                       pad=0.25,
                                                       edgecolor="blue",
                                                       facecolor="white"),
                                             font_color='k',
                                             font_size=12,
                                             )
        return nodelabel if draw else info

