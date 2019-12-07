import os
import csv
import numpy as np
import pandas as pd


class Project2:

    def __init__(self):
        self.dataset_path = 'datasets'

    def run(self):
        # The main loop of the program
        for folder in os.listdir(self.dataset_path):
            print('Processing dataset ' + folder)
            y_data, data_length = self.read_data(self.dataset_path + '/' + folder + '/targets.csv')
            y_lower = y_data['min.log.penalty']
            y_upper = y_data['max.log.penalty']
            data_length = data_length

            for loss_type in range(1, 3):
                moves_list = self.calculate_moves(y_lower, y_upper, data_length, 0, loss_type)
                self.save_result(moves_list, loss_type)

    @staticmethod
    # Dataset reader method
    def read_data(file_path):
        targets_data = pd.read_csv(file_path)
        targets_data.values.astype(np.float)
        return targets_data.to_dict(), targets_data.shape[0]

    @staticmethod
    # Save the calculated results
    def save_result(moves_list, loss_type):
        if not os.path.exists('result.csv'):
            with open('result.csv', 'w', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, ['number', 'loss_type', 'max', 'average'])
                csv_writer.writeheader()
        stat = {
            'max': 0,
            'sum': 0,
            'number': len(moves_list)
        }
        loss = 'linear'
        if loss_type != 1:
            loss = 'square'
        for element in moves_list:
            if element > stat['max']:
                stat['max'] = element
            stat['sum'] += element
        data = {
            'number': stat['number'],
            'loss_type': loss,
            'max': stat['max'],
            'average': stat['sum'] / stat['number']
        }
        with open('result.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, ['number', 'loss_type', 'max', 'average'])
            csv_writer.writerow(data)

    # The method to calculate the pointer move times
    def calculate_moves(self, y_lower, y_upper, data_length, margin, loss_type):
        # Initialize breakpoint map, pointer, function M, and moves list
        breakpoint_map = [
            [float('-inf'), {'quadratic': 0, 'linear': 0, 'constant': 0}],
            [float('inf'), {'quadratic': 0, 'linear': 0, 'constant': 0}]
        ]
        pointer = float('inf')
        function_piece = {'quadratic': 0, 'linear': 0, 'constant': 0}
        moves_list = [0 for col in range(data_length)]
        # Traverse all the data points
        for i in range(data_length):
            for s in [-1, 1]:
                # If point is lower boundary, s = -1, otherwise s = 1
                if s == -1:
                    y = y_lower[i]
                else:
                    y = y_upper[i]
                # Calculate hinge loss function at the point
                function = self.calculate_coefficients(y, margin, s, loss_type)
                # Calculate the breakpoint for the function
                break_point = y - s * margin
                breakpoint_exist_flag = False
                for index in range(len(breakpoint_map)):
                    if breakpoint_map[index][0] == break_point:
                        breakpoint_exist_flag = True
                        break
                # Add point to the map if the point is not exist
                if not breakpoint_exist_flag:
                    breakpoint_map.append([break_point, {
                        'quadratic': function['quadratic'] * s,
                        'linear': function['linear'] * s,
                        'constant': function['constant'] * s
                    }])
                    breakpoint_map.sort()
                # Update the function M
                if 0 < s * (pointer - y) + margin:
                    function_piece['quadratic'] += function['quadratic']
                    function_piece['linear'] += function['linear']
                    function_piece['constant'] += function['constant']
                # Move the pointer if the pointer is not at the min cost position
                pointer_temp_list = []
                while not self.check_min_in_interval(function_piece, breakpoint_map, pointer):
                    if pointer in pointer_temp_list:
                        break
                    pointer_temp_list.append(pointer)
                    if pointer == float('inf'):
                        _pointer = 999999999999999999
                    elif pointer == float('-inf'):
                        _pointer = -999999999999999999
                    else:
                        _pointer = pointer
                    # Check the slope of the pointer position
                    if 2 * function_piece['quadratic'] * _pointer + function_piece['linear'] > 0:
                        index = 0
                        for point in breakpoint_map:
                            if point[0] == pointer:
                                index -= 1
                                pointer = breakpoint_map[index][0]
                                break
                            else:
                                index += 1
                        function_piece['quadratic'] -= breakpoint_map[index][1]['quadratic']
                        function_piece['linear'] -= breakpoint_map[index][1]['linear']
                        function_piece['constant'] -= breakpoint_map[index][1]['constant']
                    else:
                        index = 0
                        for point in breakpoint_map:
                            if point[0] == pointer:
                                break
                            else:
                                index += 1
                        function_piece['quadratic'] += breakpoint_map[index][1]['quadratic']
                        function_piece['linear'] += breakpoint_map[index][1]['linear']
                        function_piece['constant'] += breakpoint_map[index][1]['constant']
                        pointer = breakpoint_map[index + 1][0]
                    moves_list[i] += 1
        return moves_list

    # Method of calculate the hinge loss function of the data point
    @staticmethod
    def calculate_coefficients(y, margin, s, loss_type):
        if loss_type == 1:  # Linear
            return {'quadratic': 0, 'linear': s, 'constant': margin - s * y}
        else:  # Square
            return {'quadratic': 1, 'linear': 2 * margin * s - 2 * y,
                    'constant': -2 * margin * s * y + y * y + margin * margin}

    # Method to judge whether the pointer is at the min cost position
    @staticmethod
    def check_min_in_interval(function, breakpoint_map, pointer):
        if function['quadratic'] == 0 and function['linear'] == 0:
            return True
        breakpoint_map_length = len(breakpoint_map)
        if breakpoint_map_length == 0:
            return True
        # If the function is quadratic
        if function['quadratic'] != 0:
            if pointer == float('inf'):
                return False
            if breakpoint_map_length == 1:
                return True
            index = 0
            for i in range(breakpoint_map_length):
                if breakpoint_map[i][0] == pointer:
                    index = i
                    break
            if index == 0:
                value_at_pointer = function['quadratic'] * pointer * pointer \
                                   + function['linear'] * pointer + function['constant']
                if value_at_pointer < 0:
                    value_at_pointer = 0
                value_at_right = \
                    function['quadratic'] * breakpoint_map[1][0] * breakpoint_map[1][0] \
                    + function['linear'] * breakpoint_map[1][0] + function['constant']
                if value_at_right < 0:
                    value_at_right = 0
                if value_at_right < value_at_pointer:
                    return False
                else:
                    return True
            elif index == breakpoint_map_length - 1:
                value_at_pointer = function['quadratic'] * pointer * pointer \
                                   + function['linear'] * pointer + function['constant']
                if value_at_pointer < 0:
                    value_at_pointer = 0
                value_at_left = \
                    function['quadratic'] * breakpoint_map[index - 1][0] * breakpoint_map[index - 1][0] \
                    + function['linear'] * breakpoint_map[index - 1][0] + function['constant']
                if value_at_left < 0:
                    value_at_left = 0
                if value_at_left < value_at_pointer:
                    return False
                else:
                    return True
            else:
                value_at_left = \
                    function['quadratic'] * breakpoint_map[index - 1][0] * breakpoint_map[index - 1][0] \
                    + function['linear'] * breakpoint_map[index - 1][0] + function['constant']
                if value_at_left < 0:
                    value_at_left = 0
                value_at_pointer = \
                    function['quadratic'] * pointer * pointer \
                    + function['linear'] * pointer + function['constant']
                if value_at_pointer < 0:
                    value_at_pointer = 0
                value_at_right = \
                    function['quadratic'] * breakpoint_map[index + 1][0] * breakpoint_map[index + 1][0] \
                    + function['linear'] * breakpoint_map[index + 1][0] + function['constant']
                if value_at_right < 0:
                    value_at_right = 0
                if value_at_left < value_at_pointer or value_at_right < value_at_pointer:
                    return False
                else:
                    return True
        # If the function is linear
        elif function['linear'] != 0:
            if pointer == float('inf'):
                return False
            if function['linear'] * pointer + function['constant'] <= 0:
                return True
            else:
                return False
        # If the function is constant
        else:
            return True


if __name__ == '__main__':
    Project = Project2()
    Project.run()
