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
import math
import traceback
from .const import RZ_PERIOD, RZ_PARAMS, SIMPLIFY_RULE


class QCIS_Simplify:
    """
    The class to simplify qcis code.
    """

    def __init__(self):
        self.simplify_rule = SIMPLIFY_RULE
        self.check_rule_list = [
            "rz_param_conversion",
            "gate_param_conversion",
            "gate_conversion",
        ]
        self.rz_period = RZ_PERIOD
        self.rz_params = RZ_PARAMS

    def find_qubit_by_qasm(self, conversion_value):
        """Finds qubit indices from a QASM conversion value.

        Parses the given `conversion_value` to extract qubit indices enclosed in square brackets.
        Returns a list of unique qubit indices.

        Args:
            conversion_value (str): A string containing the QASM conversion value with qubit indices
                                    represented as "[index]".

        Returns:
            list: A list of unique integers representing the qubit indices found in the `conversion_value`.
        """
        params_idx = re.findall(r"\[(\d+)\]", conversion_value)
        params_idx = list(set(params_idx))
        return params_idx

    def replace_conversion(self, rz_pi_flag=False):
        """
        Replace RZ gates with equivalent gates to simplify the circuit.

        Args:
            rz_pi_flag (bool): Flag indicating whether to replace RZ(π) with RZ(-π).

        """
        # 根据rz_param_conversion替换现有的qics，完成rz部分化简
        # gate_param_conversion gate_conversion都是类似的
        if rz_pi_flag is False:
            check_rule_list = self.check_rule_list
        else:
            check_rule_list = ["rz_param_as_pi"]
        for rule in check_rule_list:
            for conversion_key, conversion_value in self.simplify_rule[rule].items():
                pattern = re.compile(conversion_key)
                matches = pattern.finditer(self.qcis_instr)
                params_idx = self.find_qubit_by_qasm(conversion_value)
                for match in matches:
                    new_string = conversion_value
                    full_match = match.group(0)
                    # 替换化简
                    for idx in params_idx:
                        new_string = new_string.replace(
                            f"[{idx}]", match.group(int(idx))
                        )

                    # rz_param_as_pi转换后可能需要添加\n
                    if rz_pi_flag:
                        has_newline = bool(match.group(2))
                        has_newline = "\n" if has_newline else ""
                        new_string = f"{new_string}{has_newline}"
                    self.qcis_instr = self.qcis_instr.replace(full_match, new_string)

    def check_conversion(self):
        """Check whether rz_param_conversion needed.

        Iterates through each rule in the check rule list. For each rule, it contains
        a set of conversion keywords. If any keyword from the rules is found in the
        instruction, it indicates that conversion is required.

        Returns:
            bool: True if the instruction requires conversion; otherwise, False.
        """
        # 检查是否需要rz_param_conversion化简，一旦检查出有匹配的立马返回，进行化简
        # 如果都没有匹配的，返回False进行下一步检查
        for rule in self.check_rule_list:
            for conversion_key, _ in self.simplify_rule[rule].items():
                pattern = re.compile(conversion_key)
                matches = pattern.finditer(self.qcis_instr)
                if len(list(matches)) > 0:
                    return True
        return False

    # Check if optimization can continue
    def check_optimization_continue(self):
        """
        Check whether the optimization process should proceed.

        Returns:
            bool: True if the optimization should proceed, False otherwise.
        """
        flag = self.check_conversion()
        if flag:
            self.replace_conversion()
        return flag

    def repeat_rz(self):
        """
        Merge consecutive RZ gate operations.

        This method iterates through quantum instruction string, replacing RZ parameters
        with their actual values. It then identifies and merges consecutive RZ gates
        to simplify the circuit and potentially reduce execution time.

        Attributes:
            rz_params (dict): A dictionary containing parameters to be replaced in RZ instructions.
            qcis_instr (str): The quantum circuit instruction string where replacements and merges occur.
            rz_period (float): The period used for modulo operations to normalize RZ parameters within the range [-π, π].
        """
        # 将rz_params参数替换成具体值
        for param_key, param_value in self.rz_params.items():
            self.qcis_instr = self.qcis_instr.replace(param_key, str(param_value))
        # 处理重复的RZ
        # 定义正则表达式模式
        pattern = re.compile(r"(RZ Q(\d+) ([^\n]+)\n)(RZ Q\2 ([^\n]+)\n?)+")

        # 查找匹配的连续RZ门，合并参数
        matches = pattern.finditer(self.qcis_instr)
        for match in matches:
            full_match = match.group(0)
            q_number = match.group(2)
            parameters = match.group(3).split()
            # 计算连续参数的和
            parameters = [r.split(" ")[-1] for r in full_match.split("\n") if r]
            total_parameter = sum(float(param) for param in parameters)

            if total_parameter == -math.pi:
                total_parameter = math.pi
            if total_parameter > math.pi or total_parameter < -math.pi:
                total_parameter = total_parameter % (self.rz_period)
                # 如果取模2π的结果大于π，再进行-2π取模
                if total_parameter > math.pi:
                    total_parameter = total_parameter % (-self.rz_period)

            # 构建新的RZ字符串
            new_rz = f"RZ Q{q_number} {round(total_parameter,6)}\n"
            # 替换原始字符串中的匹配部分
            self.qcis_instr = self.qcis_instr.replace(full_match, new_rz, 1)

    def simplify(self, qcis):
        """Simplifies the given quantum circuit instruction string.

        This method attempts to simplify the quantum circuit instructions by replacing specific gate operations
        and optimizing circuit structure. If an exception occurs during simplification, it prints the error message
        along with the traceback and returns the original instruction string.

        Args:
            qcis (str): The quantum circuit instruction string to be simplified.

        Returns:
            str: The simplified quantum circuit instruction string.
        """
        try:
            self.qcis_instr = "\n".join([q.strip() for q in qcis.split("\n")])
            # 线路化简一开始需要先替换线路中RZ(π)
            # RZ(-π) -- > RZ(π) -- > RZ(π/2)RZ(π/2)
            # 因为RZ部分化简都基于参数为π/2来化简的
            self.replace_conversion(rz_pi_flag=True)
            while True:
                is_simplify = self.check_optimization_continue()
                # 没有能继续优化了的，返回最终的qcis
                if not is_simplify:
                    self.repeat_rz()
                    return self.qcis_instr
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return qcis
