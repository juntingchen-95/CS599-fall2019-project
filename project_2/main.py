import numpy
import pandas
import os
import csv
import math


class Project2:

    # Initialize variables
    def __init__(self):
        self.y_lower = []
        self.y_upper = []
        self.data_length = 0
        self.moves_list = []
        self.breakpoint_list = []
        self.min_pointer = 0
        self.min_coefficients = (0, 0, 0)
        self.loss_type = 1

    # The main loop
    def run(self):
        dataset_path = 'datasets'
        # Write result CSV header
        self.save_result(['number', 'loss_type', 'max', 'average', 'q1', 'q3'])
        # Read each dataset
        for folder in os.listdir(dataset_path):
            print('Process data ' + folder)
            y_data, data_length = self.read_data(dataset_path + '/' + folder + '/targets.csv')
            self.y_lower = y_data['min.log.penalty']
            self.y_upper = y_data['max.log.penalty']
            self.data_length = data_length

            for i in range(1, 3):
                self.moves_list = [0 for col in range(self.data_length)]
                self.breakpoint_list = []
                self.min_pointer = 0
                self.min_coefficients = (0, 0, 0)

                # Calculate moves in linear/square loss
                self.loss_type = i
                self.calculate_moves()
                self.save_result()

    @staticmethod
    # Dataset reader method
    def read_data(file_path):
        targets_data = pandas.read_csv(file_path)
        targets_data.values.astype(numpy.float)
        return targets_data.to_dict(), targets_data.shape[0]

    # Count and save moves
    def save_result(self, data=None):
        if data is None:
            stat = {
                'max': 0,
                'sum': 0,
                'number': len(self.moves_list)
            }
            loss_list = {
                1: 'Linear',
                2: 'Square'
            }
            for element in self.moves_list:
                if element > stat['max']:
                    stat['max'] = element
                stat['sum'] += element
            data = [
                stat['number'],
                loss_list[self.loss_type],
                stat['max'],
                stat['sum'] / stat['number'],
                self.calcualte_quantile(self.moves_list, 0.25),
                self.calcualte_quantile(self.moves_list, 0.75)
            ]
        with open('result.csv', 'a', newline='') as file:
            result_csv = csv.writer(file)
            result_csv.writerow(data)

    # Method to calculate pointer moves
    def calculate_moves(self):
        for i in range(self.data_length):
            if self.y_lower[i] == self.y_upper[i]:
                if float('-inf') < self.y_lower[i]:
                    coefficients_1 = self.calculate_coefficients(self.y_lower[i], 0., -1, self.loss_type)
                    coefficients_2 = self.calculate_coefficients(self.y_upper[i], 0., 1, self.loss_type)
                    self.moves_list[i] += self.insert_break_point(self.y_lower[i], coefficients_1, -1) + \
                                          self.insert_break_point(self.y_upper[i], coefficients_2, 1)
            else:
                if float('-inf') < self.y_lower[i]:
                    coefficient = self.calculate_coefficients(self.y_lower[i], 1, -1, self.loss_type)
                    self.moves_list[i] += self.insert_break_point(self.y_lower[i] + 1, coefficient, -1)
                if self.y_upper[i] < float('inf'):
                    coefficient = self.calculate_coefficients(self.y_upper[i], 1, 1, self.loss_type)
                    self.moves_list[i] += self.insert_break_point(self.y_upper[i] - 1, coefficient, 1)
        return

    @staticmethod
    def calculate_coefficients(y, margin, s, loss_type):
        if loss_type == 1:  # Linear
            return 0, s, margin - s * y
        else:  # Square
            return 1, 2 * margin * s - 2 * y, -2 * margin * s * y + y * y + margin * margin

    # Method to calculate moves for each pointer
    def insert_break_point(self, point, coefficients, s):
        pointer_moves = 0
        _coefficients = (coefficients[0] * s, coefficients[1] * s, coefficients[2] * s)
        if len(self.breakpoint_list) == 0:
            self.breakpoint_list.append([point, _coefficients])
            if s == 1:
                self.min_pointer = point
                pointer_moves += 1
        else:
            added_flag = False
            for i in range(len(self.breakpoint_list)):
                if self.breakpoint_list[i][0] == point:
                    added_flag = True
                    # If point is exist, renew the coefficients
                    self.breakpoint_list[i][1] = (
                        coefficients[0] + _coefficients[0], coefficients[1] + _coefficients[1],
                        coefficients[2] + _coefficients[2])
                    break
            if not added_flag:
                # If point is not exist, insert the point and coefficients
                self.breakpoint_list.append([point, _coefficients])
                self.breakpoint_list.sort()

            # Get min pointer and coefficient
            min_breakpoint_position = self.get_breakpoint_position(self.min_pointer)
            if self.position_check(point, s, min_breakpoint_position):
                self.min_coefficients = (
                    self.min_coefficients[0] + coefficients[0], self.min_coefficients[1] + coefficients[1],
                    self.min_coefficients[2] + coefficients[2])

            # Move the min pointer to the left
            if self.check_increase_or_decrease(self.min_coefficients, min_breakpoint_position, 1):
                while self.min_pointer != self.breakpoint_list[0][0] and \
                        not self.check_increase_or_decrease(self.min_coefficients, min_breakpoint_position, 0) and \
                        not self.check_min_interval(self.min_coefficients,
                                                self.get_breakpoint_position(self.get_previous_point(self.min_pointer)),
                                                self.min_pointer):
                    self.min_pointer = self.get_previous_point(self.min_pointer)
                    min_breakpoint_position = self.get_breakpoint_position(self.min_pointer)
                    self.min_coefficients = self.coefficient_subtraction(self.min_coefficients, self.min_pointer)
                    pointer_moves += 1

            # Move the min pointer to the right
            elif self.check_increase_or_decrease(self.min_coefficients, min_breakpoint_position, 0):
                while self.min_pointer != self.breakpoint_list[len(self.breakpoint_list) - 1][0] and \
                        (not self.check_increase_or_decrease(
                            self.coefficient_addition(self.min_coefficients, self.min_pointer),
                            self.get_breakpoint_position(self.get_next_point(self.min_pointer)), 1) or
                         self.check_min_interval(self.coefficient_addition(self.min_coefficients, self.min_pointer),
                                                 self.get_breakpoint_position(self.min_pointer),
                                                 self.get_breakpoint_position(self.get_next_point(self.min_pointer)))):
                    self.min_coefficients = self.coefficient_addition(self.min_coefficients, self.min_pointer)
                    self.min_pointer = self.get_next_point(self.min_pointer)
                    min_breakpoint_position = self.get_breakpoint_position(self.min_pointer)
                    pointer_moves += 1
        return pointer_moves

    def get_breakpoint_position(self, point):
        if point == self.breakpoint_list[len(self.breakpoint_list) - 1][0]:
            return float('inf')
        else:
            return point

    @staticmethod
    def position_check(break_point, s, break_position):
        if s == 1 and break_position > break_point:
            return True
        elif s == -1 and break_position <= break_point:
            return True
        else:
            return False

    @staticmethod
    # Check the position is increase or decrease
    def check_increase_or_decrease(coefficients, value, mode):
        gradient = coefficients[0] * 2 * value + coefficients[1]
        if mode == 0:
            if gradient < 0:
                return True
        else:
            if gradient > 0:
                return True
        return False

    @staticmethod
    def check_min_interval(coefficients, value_1, value_2):
        if coefficients[0] == 0 and coefficients[1] == 0:
            return True
        else:
            if coefficients[0] != 0:
                min_value = -coefficients[1] / (2 * coefficients[0])
            elif coefficients[1] != 0:
                min_value = float('-inf') * coefficients[1]
            else:
                min_value = 0
            return value_1 < min_value < value_2 or value_2 == min_value

    # Get the previous point of a point in the list
    def get_previous_point(self, point):
        for i in range(len(self.breakpoint_list)):
            if self.breakpoint_list[i][0] == point and i > 0:
                return self.breakpoint_list[i - 1][0]
        else:
            return self.breakpoint_list[0][0]

    # Get the next point of a point in the list
    def get_next_point(self, point):
        for i in range(len(self.breakpoint_list)):
            if self.breakpoint_list[i][0] == point and i < len(self.breakpoint_list) - 1:
                return self.breakpoint_list[i + 1][0]
        else:
            return self.breakpoint_list[len(self.breakpoint_list) - 1][0]

    def coefficient_addition(self, coefficient, point):
        for breakpoint_element in self.breakpoint_list:
            if breakpoint_element[0] == point:
                return (
                    coefficient[0] + breakpoint_element[1][0],
                    coefficient[1] + breakpoint_element[1][1],
                    coefficient[2] + breakpoint_element[1][2]
                )
        return coefficient

    def coefficient_subtraction(self, coefficient, point):
        for breakpoint_element in self.breakpoint_list:
            if breakpoint_element[0] == point:
                return (
                    coefficient[0] - breakpoint_element[1][0],
                    coefficient[1] - breakpoint_element[1][1],
                    coefficient[2] - breakpoint_element[1][2]
                )
        return coefficient

    @staticmethod
    def calcualte_quantile(data, p):
        data.sort()
        pos = (len(data) + 1) * p
        pos_integer = int(math.modf(pos)[1])
        pos_decimal = pos - pos_integer
        return data[pos_integer - 1] + (data[pos_integer] - data[pos_integer - 1]) * pos_decimal


if __name__ == '__main__':
    Project = Project2()
    Project.run()
