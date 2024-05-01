import openpyxl
# from scipy.optimize import linprog
import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds
from math import ceil
import time
from sys import argv
import json
# from sys import stdin


def matrix_array(sheet, row: int, column: int, m: int, n: int) -> list[list[int]]:
    arr = []
    for i in range(m):
        arr.append([])
        for j in range(n):
            arr[i].append(sheet.cell(row + i, column + j).value)
    return arr


def read_test_data_from_excel(path: str):
    space_between_tables_below = 3
    space_between_tables_right = 1
    first_tables_row = 8
    first_tables_column = 3
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    m = sheet.cell(3, 2).value
    n = sheet.cell(3, 3).value
    model = {}
    table_names = ["c_car", "c_rail", "c_plane", "gk_car", "gk_rail", "gk_plane", "t_car", "t_rail", "t_plane"]

    for index in range(len(table_names)):
        match index + 1:
            case 1:
                current_row = first_tables_row
                current_column = first_tables_column
            case 4:
                current_row = first_tables_row + m + space_between_tables_below + 1
                current_column = first_tables_column
            case 7:
                current_row = first_tables_row + 2 * (m + space_between_tables_below) + 2
                current_column = first_tables_column
            case _:
                current_column += n + space_between_tables_right + 1

        model[table_names[index]] = matrix_array(sheet, current_row, current_column, m, n)

    stocks = []
    current_row += space_between_tables_below + m
    current_column = first_tables_column + n

    for i in range(m):
        current_row += 1
        stocks.append(sheet.cell(current_row, current_column).value)

    current_column = first_tables_column - 1
    current_row += 1
    needs = []
    for i in range(n):
        current_column += 1
        needs.append(sheet.cell(current_row, current_column).value)

    model['stocks'] = stocks
    model['needs'] = needs
    model['time'] = sheet.cell(3, 9).value
    return model


def fill_optimization_func(c: list[list[float]], gk: list[list[float]], transport_name: str, d_model: dict,
                           time: float) -> dict:
    last_c_index = d_model['last_c_index']
    b_u = d_model['b_u']
    b_l = d_model['b_l']
    opt = d_model['opt']
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    lb = d_model['lb']
    ub = d_model['ub']
    integrality = d_model['integrality']
    check_time = False
    for row in range(len(c)):
        for column in range(len(c[0])):
            if c[row][column] != None:
                check_time = True
                v_index[f'r_{row}_{column}_{transport_name}'] = len(opt)
                opt.append(c[row][column] * gk[row][column])
                ub.append(np.inf)
                lb.append(0)
                # lb.append(0.1)
                integrality.append(0)
                v_index[f'w_{row}_{column}_{transport_name}'] = len(opt)
                opt.append(0)
                ub.append(1)
                lb.append(0)
                integrality.append(1)

                last_c_index += 1
                c_index[f'c_gw_r_{row}_{column}_{transport_name}'] = last_c_index
                b_u.append(np.inf)
                b_l.append(0)
                last_c_index += 1
                c_index[f'c_w_r_{row}_{column}_{transport_name}'] = last_c_index
                b_u.append(0)
                b_l.append(-np.inf)
        if (check_time):
            check_time = False
            last_c_index += 1
            c_index[f'c_s_{row}_{transport_name}'] = last_c_index
            b_u.append(time)
            b_l.append(-np.inf)
    return {'opt': opt, 'v_index': v_index, 'c_index': c_index, 'b_u': b_u, 'b_l': b_l, 'ub': ub, 'lb': lb,
            'integrality': integrality, 'last_c_index': last_c_index}


def fill_matrix_array(matrix_array: list[list[float]], c: list[list[float]],
                      gk: list[list[float]], t: list[list[float]],
                      v_index: dict, c_index: dict, transport_name: str, big_number: int) -> list[list[float]]:
    for row in range(len(c)):
        for column in range(len(c[0])):
            if c[row][column] != None:
                i = c_index[f'c_a{row}']
                j_r = v_index[f'r_{row}_{column}_{transport_name}']
                matrix_array[i][j_r] = gk[row][column]

                i = c_index[f'c_b{column}']
                matrix_array[i][j_r] = gk[row][column]

                i = c_index[f'c_s_{row}_{transport_name}']
                j_w = v_index[f'w_{row}_{column}_{transport_name}']
                matrix_array[i][j_r] = t[row][column]
                matrix_array[i][j_w] = t[row][column]

                i = c_index[f'c_gw_r_{row}_{column}_{transport_name}']
                matrix_array[i][j_r] = -1
                matrix_array[i][j_w] = big_number

                i = c_index[f'c_w_r_{row}_{column}_{transport_name}']
                matrix_array[i][j_r] = -1
                matrix_array[i][j_w] = 1
    return matrix_array


def solve_problem(model: dict):
    # opt = []
    # d_index = {}
    d_model = {'opt': [], 'v_index': {}, 'c_index': {}, 'ub': [], 'lb': [], 'b_u': [], 'b_l': [], 'integrality': [],
               'last_c_index': -1}
    s = model['time']
    d_model = fill_optimization_func(model['c_car'], model['gk_car'], 'car', d_model, s)
    d_model = fill_optimization_func(model['c_rail'], model['gk_rail'], 'rail', d_model, s)
    d_model = fill_optimization_func(model['c_plane'], model['gk_plane'], 'plane', d_model, s)

    last_c_index = d_model['last_c_index']
    c_index = d_model['c_index']
    b_u = d_model['b_u']
    b_l = d_model['b_l']
    needs = model['needs']
    stocks = model['stocks']
    # print(needs)
    for i in range(len(model['c_car'])):
        last_c_index += 1
        c_index[f'c_a{i}'] = last_c_index
        b_u.append(stocks[i])
        b_l.append(-np.inf)
    for i in range(len(model['c_car'][0])):
        last_c_index += 1
        c_index[f'c_b{i}'] = last_c_index
        b_u.append(needs[i])
        b_l.append(needs[i])
    matrix_array = []
    for i in range(len(d_model['c_index'])):
        matrix_array.append([])
        for j in range(len(d_model['opt'])):
            matrix_array[i].append(0)
    matrix_array = fill_matrix_array(matrix_array, model['c_car'], model['gk_car'], model['t_car'], d_model['v_index'],
                                     d_model['c_index'], 'car', 1000)
    matrix_array = fill_matrix_array(matrix_array, model['c_rail'], model['gk_rail'], model['t_rail'],
                                     d_model['v_index'], d_model['c_index'], 'rail', 1000)
    matrix_array = fill_matrix_array(matrix_array, model['c_plane'], model['gk_plane'], model['t_plane'],
                                     d_model['v_index'], d_model['c_index'], 'plane', 1000)
    c = np.array(d_model['opt'])
    cons = LinearConstraint(np.array(matrix_array), np.array(d_model['b_l']), np.array(d_model['b_u']))
    inter = np.array(d_model['integrality'])
    bounds = Bounds(np.array(d_model['lb']), np.array(d_model['ub']))

    start_time = time.perf_counter()
    res = milp(c=c, constraints=cons, integrality=inter, bounds=bounds)
    end_time = time.perf_counter()

    return res, end_time - start_time, d_model['v_index'], len(np.array(matrix_array))


def convert_res(res, v_index: dict, model) -> dict:
    # opt: list[float]
    d_res = {}
    additional_info = {'missing_values': [], 'c_a': {}, 'c_b': {}, 'c_s': {}}
    missing_values = additional_info['missing_values']
    c_a = additional_info['c_a']
    c_b = additional_info['c_b']
    c_s = additional_info['c_s']

    for i in range(len(res.x)):
        if res.x[i] != 0 and round(res.x[i], 3) != 0:
            name = [k for k, v in v_index.items() if v == i][0]
            temp = name.split('_')
            s_old = f'_{temp[1]}_{temp[2]}_{temp[3]}'
            s_new = f'_{int(temp[1]) + 1}_{int(temp[2]) + 1}_{temp[3]}'

            # d_res[temp[0]+s_new] = res.x[i]

            if temp[0] == 'r':
                # d_res[temp[0]+s_new] = round(res.x[i],3)
                d_res['x' + s_new] = round(res.x[i] * model[f'gk_{temp[3]}'][int(temp[1])][int(temp[2])], 3)

                if c_a.get(f'c_a{int(temp[1]) + 1}') == None:
                    c_a[f'c_a{int(temp[1]) + 1}'] = model[f'gk_{temp[3]}'][int(temp[1])][int(temp[2])] * res.x[i]
                else:
                    c_a[f'c_a{int(temp[1]) + 1}'] += model[f'gk_{temp[3]}'][int(temp[1])][int(temp[2])] * res.x[i]

                if c_b.get(f'c_b{int(temp[2]) + 1}') == None:
                    c_b[f'c_b{int(temp[2]) + 1}'] = model[f'gk_{temp[3]}'][int(temp[1])][int(temp[2])] * res.x[i]
                else:
                    c_b[f'c_b{int(temp[2]) + 1}'] += model[f'gk_{temp[3]}'][int(temp[1])][int(temp[2])] * res.x[i]

                if c_s.get(f'c_s_{int(temp[1]) + 1}_{temp[3]}') == None:
                    c_s[f'c_s_{int(temp[1]) + 1}_{temp[3]}'] = model[f't_{temp[3]}'][int(temp[1])][int(temp[2])] * ceil(
                        res.x[i])
                else:
                    c_s[f'c_s_{int(temp[1]) + 1}_{temp[3]}'] += model[f't_{temp[3]}'][int(temp[1])][
                                                                    int(temp[2])] * ceil(res.x[i])

                if res.x[v_index.get('w' + s_old)] == 0:
                    missing_values.append('w' + s_new)
            else:
                if res.x[v_index.get('r' + s_old)] == 0:
                    missing_values.append('r' + s_new)

    return d_res, additional_info


# path = '/content/test_data_2x2.xlsx'
path = "E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\test_data\\test_data_2x2.xlsx"
# path = "E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\test_data\\test_data_20x40.xlsx"
model = read_test_data_from_excel(path)
res, working_time, v_index, cons_len = solve_problem(model)
d_res, additional_info = convert_res(res, v_index, model)

for pair in d_res.items():
    print(pair)
print("result = ", res.fun)
print("working_time = ", working_time)
print("nodes count = ", res.mip_node_count)
print("variables count = ", len(res.x))
print("constraints count = ", cons_len)
print("missing values = ", additional_info['missing_values'])
print('restrictions on a:', additional_info['c_a'])
print('restrictions on b:', additional_info['c_b'])
print('restrictions on s:', additional_info['c_s'])

# script_name, first = argv
# print("script_name: ", script_name)
# print("first variable type: ", type(first), ' variable: ', first)
print("model = ", model)

# Открываем файл для чтения
print('\n\n')
with open("E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\module_data\\module_input_data.json", 'r', encoding='utf-8') as file:
    # Загружаем содержимое файла в переменную
    data = json.load(file)

# Выводим загруженные данные
print("data = ", data)
print("data_type = ", type(data))
# print("data_el = ", data['t_car'])
