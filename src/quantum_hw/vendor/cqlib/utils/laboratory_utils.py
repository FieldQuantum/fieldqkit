# This code is part of cqlib.
#
# Copyright (C) 2024 China Telecom Quantum Group, QuantumCTek Co., Ltd.,
# Center for Excellence in Quantum Information and Quantum Physics.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import re

import numpy as np
from .qasm_to_qcis.qasm_to_qcis import QasmToQcis
from .qcis_to_qasm import QcisToQasm
from typing import List, Optional, Dict, Union
from .simplify import QCIS_Simplify


class LaboratoryUtils:

    def __init__(self, machine_name: Optional[str] = None):
        """account initialization

        Args:
            machine_name:
                name of quantum computer. Defaults to None.

        Raises:
            Exception: throw an exception when login fails
        """
        self.qasmtoqcis = QasmToQcis()
        self.qcistoqasm = QcisToQasm()
        self.qcis_simplify = QCIS_Simplify()
        self.machine_name = machine_name

    def assign_parameters(
            self, circuits: List[str], parameters: List[List], values: List[List]
    ):
        """
        Assign parameter values to circuit template strings based on provided parameter lists.

        Args:
            circuits (List[str]): A list of circuit template QCIS strings which may contain parameter placeholders.
            parameters (List[List]): A list of parameter names corresponding to the `values` list.
            values (List[List]): A list of parameter values corresponding to the `parameters` list.

        Returns:
            List[str]: A list containing each circuit template string after parameter substitution.
                       Returns an empty string if parameter matching fails or the number of parameters mismatches.
        """
        new_circuit = []
        for circuit, parameter, value in zip(circuits, parameters, values):
            circuit = circuit.upper()
            p = re.compile(r"\{(\w+)\}")
            circuit_parameters = p.findall(circuit)
            if circuit_parameters:

                # 将所有parameter变为大写， 否则set(parameters) != set(circuit_parameters) 不通过
                after_parameter = [p.upper() for p in parameter]

                if not value:
                    error_message = (
                        f"线路含有参数{circuit_parameters}, 请提供相应的参数值"
                    )
                    print(error_message)
                    return ""

                else:
                    if len(circuit_parameters) != len(value):
                        error_message = f"线路含有{len(circuit_parameters)}个参数, 您提供了{len(value)}个参数值"
                        print(error_message)
                        return ""

                    elif after_parameter and len(circuit_parameters) != len(
                            after_parameter
                    ):
                        error_message = f"线路含有{len(circuit_parameters)}个参数, 您提供了{len(after_parameter)}个参数"
                        print(error_message)
                        return ""

                    elif set(after_parameter) != set(circuit_parameters):
                        error_message = "线路中的参数与您输入的参数名称不符"
                        print(error_message)
                    else:
                        param_dic = {}
                        for p, v in zip(after_parameter, value):
                            param_dic[p] = v
                        expData = circuit.format(**param_dic)
                        new_circuit.append(expData)
            elif parameter or value:
                error_message = "线路定义中不含有参数，无法接受您输入的参数或参数值"
                print(error_message)
                return ""
            else:
                expData = circuit
                new_circuit.append(expData)
        return new_circuit

    def convert_qasm_to_qcis(self, qasm: str, qubit_map: Optional[Dict] = None):
        """Convert qasm string to qcis string

        Args:
             qasm:
                 qasm string.
             qubit_map:
                 Number mapping in qasm, where the value is None,
                 directly maps bits based on the format of number plus 1. Defaults to None.

         Raises:
             Exception: language conversion failed.

         Returns:
             str: simplified qcis string.
        """
        qcis_raw = self.qasmtoqcis.convert_qasm_to_qcis(qasm, qubit_map=qubit_map)
        simplity_qcis = self.simplify_qcis(qcis_raw)
        return simplity_qcis

    def convert_qasm_to_qcis_from_file(
            self, qasm_file: str, qubit_map: Optional[Dict] = None
    ):
        """Read qasm from file and convert it to qcis

        Args:
            qasm_file:
                qasm file.
            qubit_map:
                Number mapping in qasm, where the value is None,
                directly maps bits based on the format of number plus 1. Defaults to None.

        Raises:
            Exception: language conversion failed.

        Returns:
            str: simplified qcis.
        """
        qcis_raw = self.qasmtoqcis.convert_qasm_to_qcis_from_file(
            qasm_file, qubit_map=qubit_map
        )
        simplity_qcis = self.simplify_qcis(qcis_raw)
        return simplity_qcis

    def convert_qcis_to_qasm(self, qcis: str):
        """convert qcis to qasm string.

        Args:
            qcis: qcis

        Returns:
            str: converted qasm.
        """
        qasm_circuit = self.qcistoqasm.convert_qcis_to_qasm(qcis)
        return qasm_circuit

    def simplify_qcis(self, qcis_raw: str):
        """simplification of qcis lines.
        If simplification fails, prompt an error message and return the original qcis circuit.

        Args:
            qcis_raw: qcis

        Returns:
            str: simplified qcis.
        """
        simplity_qcis = self.qcis_simplify.simplify(qcis_raw)
        return simplity_qcis

    def readout_data_to_state_probabilities(self, result):
        """
        Converts readout data into a list of state probabilities.

        Parses the measurement results to derive the state of each qubit and
        represents it as a binary-formatted list. This function effectively
        translates the measurement outcomes into the probability of the quantum
        system being in a specific state post-measurement.

        Args:
            result (dict): A dictionary containing the measurement results, where
                           the '.resultStatus' entry is a list with the first item
                           being the number of qubits, followed by the measurement
                           outcomes for each qubit.

        Returns:
            list: A list of each qubit's states, with each qubit's state represented
                  as a boolean list; True signifies the qubit is in state 1, and False
                  indicates state 0.
        """
        state01 = result.get("resultStatus")
        basis_list = []
        basis_content = "".join(
            ["".join([str(s) for s in state]) for state in state01[1:]]
        )
        qubits_num = len(state01[0])  # 测量比特个数
        for idx in range(qubits_num):
            basis_result = basis_content[idx: len(basis_content): qubits_num]
            basis_list.append([True if res == "1" else False for res in basis_result])
        return basis_list

    # 读取数据转换成量子态概率
    def readout_data_to_state_probabilities_whole(self, result: Dict):
        """read data and convert it into a quantum state probability, all returns.

        Args:
            result: the results returned after query_experiment.

        Returns:
            Dict: probability
        """
        basis_list = self.readout_data_to_state_probabilities(result)
        probabilities = self.original_onversion_whole(basis_list)
        return probabilities

    def readout_data_to_state_probabilities_part(self, result: Dict):
        """read data and convert it into a quantum state probability, do not return with a probability of 0.

        Args:
            result: the results returned after query_experiment.

        Returns:
            Dict: probability
        """
        basis_list = self.readout_data_to_state_probabilities(result)
        probabilities = self.original_onversion_part(basis_list)
        return probabilities

    def original_onversion_whole(self, state01):
        """
        Calculates the probability distribution of measurement outcomes based on the given quantum state.

        Args:
            state01: A one-dimensional or two-dimensional list representing the quantum state, with elements as complex numbers.

        Returns:
            Dict: A dictionary with keys as binary strings of measurement results and values as corresponding probabilities.
        """
        # 当state01为一维时转换成二维数据
        if isinstance(state01[0], bool):
            state01 = [state01]
        n = len(state01)  # 读取比特数
        counts = [0] * (2 ** n)
        state01_T = np.transpose(state01)  # 转置
        numShots = len(state01_T)  # 测量重复次数
        # 统计所有numShots 列
        for num in range(numShots):
            k = 0
            for i in range(n):
                k += state01_T[num][i] * (2 ** i)
            counts[k] += 1
        # 计算概率
        P = {bin(k)[2:].zfill(n): counts[k] / numShots for k in range(2 ** n)}
        return P

    def original_onversion_part(self, state01):
        """Calculates the probability distribution of measurement outcomes for a given quantum state.

        Args:
            state01 (list): A one-dimensional or two-dimensional list representing the quantum state, with complex numbers as elements.

        Returns:
            dict: A dictionary where keys are binary strings representing measurement outcomes and values are their corresponding probabilities.
        """
        # 当state01为一维时转换成二维数据
        if isinstance(state01[0], bool):
            state01 = [state01]
        n = len(state01)  # 读取比特数
        counts = {}
        state01_T = np.transpose(state01)  # 转置
        numShots = len(state01_T)  # 测量重复次数
        # 统计所有numShots 列
        for num in range(numShots):
            k = 0
            for i in range(n):
                k += state01_T[num][i] * (2 ** i)
            prob_state = bin(k)[2:].zfill(n)
            if prob_state not in counts:
                counts[prob_state] = 1
            else:
                counts[prob_state] += 1
        # 计算概率
        P = {k: v / numShots for k, v in counts.items()}
        return P

    # 量子态概率矫正
    def probability_calibration(
            self, result: Dict, laboratory, config_json: Optional[Dict] = None
    ):
        """correction of the measured probability of 01 quantum state.

        Args:
            result:
                the results returned after query_experiment.
            config_json:
                experimental parameters of quantum computer.
                config_json value is None, read the latest experimental parameters for calculation.
                Defaults to None.
            laboratory:
        Raises:
            Exception: cannot calibrate probability with fidelity.

        Returns:
            Dict: corrected probability.
        """
        CM_CACHE = {}
        if config_json is None:
            config_json = laboratory.download_config()
        qubit_num = [f"Q{i}" for i in result.get("resultStatus")[0]]
        n = len(qubit_num)  # 测量比特个数
        qubits = config_json["readout"]["readoutArray"]["|0> readout fidelity"][
            "qubit_used"
        ]
        readout_fidelity0 = config_json["readout"]["readoutArray"][
            "|0> readout fidelity"
        ]["param_list"]
        readout_fidelity1 = config_json["readout"]["readoutArray"][
            "|1> readout fidelity"
        ]["param_list"]
        iq2probFidelity = [
            [readout_fidelity0[qubits.index(q)], readout_fidelity1[qubits.index(q)]]
            for q in qubit_num
        ]
        P = self.readout_data_to_state_probabilities_whole(result)
        Pm = list(P.values())
        if not isinstance(iq2probFidelity[0], list):
            iq2probFidelity = [iq2probFidelity]
        f = tuple([float(fi) for fi in sum(iq2probFidelity, [])])
        if f not in CM_CACHE:
            inv_CM = 1
            for k in iq2probFidelity[::-1]:
                F00 = k[0]
                F11 = k[1]
                if F00 + F11 == 1:
                    raise Exception(
                        f"Cannot calibrate probability with fidelity: [{F00}, {F11}]"
                    )
                inv_cm = np.array([[F11, F11 - 1], [F00 - 1, F00]]) / (F00 + F11 - 1)
                inv_CM = np.kron(inv_CM, inv_cm)
            CM_CACHE[f] = inv_CM
        else:
            inv_CM = CM_CACHE[f]
        Pi = np.dot(inv_CM, (np.array(Pm, ndmin=2).T))
        Pi = {bin(idx)[2:].zfill(n): k[0] for idx, k in enumerate(Pi)}
        return Pi

    # 对矫正后的概率进行修正
    def probability_correction(self, probabilities):
        """correction of the measured probability of 01 quantum state.
           If there is a probability greater than 1, change this item to 1.
           If there is anything less than 0, change the item to 0.

        Args:
            probabilities:
                corrected probability.

        Returns:
            Dict: corrected probability.
        """
        abnormal_fidelity_list = list(
            filter(lambda x: x < 0 or x > 1, probabilities.values())
        )
        if not abnormal_fidelity_list:
            return probabilities
        for k, v in probabilities.items():
            if v > 1:
                probabilities[k] = 1
            elif v < 0:
                probabilities[k] = 0
        fidelity_sum = sum(probabilities.values())
        for k, v in probabilities.items():
            probabilities[k] = v / fidelity_sum
        return probabilities

    def get_coupling_map(self, config_json):
        """Constructs an adjacency list representing the coupling map of qubits.

        Parses the configuration file to extract the overview of qubits and coupler
        information, filters out unused qubits, and generates an adjacency list
        depicting the coupling relationship between qubits.

        Args:
            config_json (dict): A dictionary containing the configuration details of the quantum processor, including qubit lists and coupling maps.

        Returns:
            adjacency_list (list): A list representing the coupling relationships between qubits as an adjacency list.
        """
        qubits = config_json["overview"]["qubits"]
        qubits_used = config_json["qubit"]["singleQubit"]["gate error"]["qubit_used"]
        disable_qubits = [q for q in qubits if q not in qubits_used]
        coupler_map = config_json["overview"]["coupler_map"]
        adjacency_list = []
        for Q1, Q2 in coupler_map.values():
            q1 = int(Q1[1:])
            q2 = int(Q2[1:])
            if Q1 in disable_qubits or Q2 in disable_qubits:
                continue
            adjacency_list.append([q1, q2])
        return adjacency_list
