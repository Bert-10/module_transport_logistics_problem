import openpyxl
# from scipy.optimize import linprog
import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds
from math import ceil
import time
from sys import argv
import json


def fill_optimization_func_frac(product, d_model: dict) -> dict:
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
    product_name = product['product_name']
    product_time = product['time']
    for transport in product['transport_list']:
        transport_name = transport['transport_name']
        cost = transport['cost']
        gk = transport['gk']
        calc_trips = transport['calc_trips']
        for row in range(len(transport['cost'])):
            stock_name = product['stocks'][row]['stock_name']
            for column in range(len(transport['cost'][0])):
                if transport['cost'][row][column] != 'nan':
                    need_name = product['needs'][column]['need_name']
                    check_time = True
                    v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"] = len(opt)
                    if calc_trips[row][column] == 1:
                        opt.append(cost[row][column])
                    else:
                        opt.append(cost[row][column] * gk[row][column])
                    ub.append(np.inf)
                    lb.append(0)
                    # lb.append(0.1)
                    integrality.append(0)
                    v_index[f'w_{product_name}_{transport_name}_{stock_name}_{need_name}'] = len(opt)
                    opt.append(0)
                    ub.append(1)
                    lb.append(0)
                    integrality.append(1)

                    last_c_index += 1
                    c_index[f'c_gw_r_{product_name}_{transport_name}_{stock_name}_{need_name}'] = last_c_index
                    b_u.append(np.inf)
                    b_l.append(0)
                    last_c_index += 1
                    c_index[f'c_w_r_{product_name}_{transport_name}_{stock_name}_{need_name}'] = last_c_index
                    b_u.append(0)
                    b_l.append(-np.inf)
            if check_time:
                check_time = False
                if f'c_s_{product["group"]}_{transport_name}_{stock_name}' not in c_index.keys():
                    last_c_index += 1
                    c_index[f'c_s_{product["group"]}_{transport_name}_{stock_name}'] = last_c_index
                    b_u.append(product_time)
                    b_l.append(-np.inf)
    d_model['last_c_index'] = last_c_index
    return d_model


def fill_optimization_func_whole(product, d_model: dict) -> dict:
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
    product_name = product['product_name']
    product_time = product['time']
    for transport in product['transport_list']:
        transport_name = transport['transport_name']
        cost = transport['cost']
        gk = transport['gk']
        calc_trips = transport['calc_trips']
        for row in range(len(transport['cost'])):
            stock_name = product['stocks'][row]['stock_name']
            for column in range(len(transport['cost'][0])):
                if transport['cost'][row][column] != 'nan':
                    need_name = product['needs'][column]['need_name']
                    check_time = True
                    v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"] = len(opt)
                    if calc_trips[row][column] == 1:
                        opt.append(cost[row][column])
                    else:
                        opt.append(cost[row][column] * gk[row][column])
                    ub.append(np.inf)
                    lb.append(0)
                    integrality.append(1)
                    v_index[f'z_{product_name}_{transport_name}_{stock_name}_{need_name}'] = len(opt)
                    if calc_trips[row][column] == 1:
                        opt.append(0)
                    else:
                        opt.append(cost[row][column])
                    ub.append(gk[row][column]-1)
                    lb.append(0)
                    integrality.append(0)

                    last_c_index += 1
                    c_index[f'c_z_r_{product_name}_{transport_name}_{stock_name}_{need_name}'] = last_c_index
                    b_u.append(0)
                    b_l.append(-np.inf)
            if check_time:
                check_time = False
                if f'c_s_{product["group"]}_{transport_name}_{stock_name}' not in c_index.keys():
                    last_c_index += 1
                    c_index[f'c_s_{product["group"]}_{transport_name}_{stock_name}'] = last_c_index
                    b_u.append(product_time)
                    b_l.append(-np.inf)
    d_model['last_c_index'] = last_c_index
    return d_model


def fill_matrix_array_frac(product: dict, matrix_array: list[list[float]], d_model: dict, big_number=1000) -> list[
    list[float]]:
    # print('product_GROUP = =', product['group'])
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    product_name = product['product_name']
    for transport in product['transport_list']:
        transport_name = transport['transport_name']
        gk = transport['gk']
        time = transport['time']
        for row in range(len(transport['cost'])):
            stock_name = product['stocks'][row]['stock_name']
            for column in range(len(transport['cost'][0])):
                if transport['cost'][row][column] != 'nan':
                    need_name = product['needs'][column]['need_name']
                    i = c_index[f'c_a_{product_name}_{stock_name}']
                    j_r = v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"]
                    matrix_array[i][j_r] = gk[row][column]

                    i = c_index[f'c_b_{product_name}_{need_name}']
                    matrix_array[i][j_r] = gk[row][column]

                    i = c_index[f'c_s_{product["group"]}_{transport_name}_{stock_name}']
                    j_w = v_index[f'w_{product_name}_{transport_name}_{stock_name}_{need_name}']
                    matrix_array[i][j_r] = time[row][column]
                    matrix_array[i][j_w] = time[row][column]

                    i = c_index[f'c_gw_r_{product_name}_{transport_name}_{stock_name}_{need_name}']
                    matrix_array[i][j_r] = -1
                    matrix_array[i][j_w] = big_number

                    i = c_index[f'c_w_r_{product_name}_{transport_name}_{stock_name}_{need_name}']
                    matrix_array[i][j_r] = -1
                    matrix_array[i][j_w] = 1
    return matrix_array


def fill_matrix_array_whole(product: dict, matrix_array: list[list[float]], d_model: dict) -> list[
    list[float]]:
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    product_name = product['product_name']
    for transport in product['transport_list']:
        transport_name = transport['transport_name']
        gk = transport['gk']
        time = transport['time']
        for row in range(len(transport['cost'])):
            stock_name = product['stocks'][row]['stock_name']
            for column in range(len(transport['cost'][0])):
                if transport['cost'][row][column] != 'nan':
                    need_name = product['needs'][column]['need_name']
                    i = c_index[f'c_a_{product_name}_{stock_name}']
                    j_r = v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"]
                    j_z = v_index[f'z_{product_name}_{transport_name}_{stock_name}_{need_name}']
                    matrix_array[i][j_r] = gk[row][column]
                    matrix_array[i][j_z] = -1

                    i = c_index[f'c_b_{product_name}_{need_name}']
                    matrix_array[i][j_r] = gk[row][column]
                    matrix_array[i][j_z] = -1

                    i = c_index[f'c_s_{product["group"]}_{transport_name}_{stock_name}']
                    matrix_array[i][j_r] = time[row][column]

                    i = c_index[f'c_z_r_{product_name}_{transport_name}_{stock_name}_{need_name}']
                    matrix_array[i][j_r] = -gk[row][column]
                    matrix_array[i][j_z] = 1
    return matrix_array


def solve_problem(data: dict) -> list[dict]:
    group_dict = {}
    for product in data['products_list']:
        if product['group'] in group_dict:
            group_dict[product['group']].append(product)
        else:
            group_dict[product['group']] = [product]
    plan_list = []
    # print(group_dict)
    for group, group_product_list in group_dict.items():
        plan_dict = {'group': group, 'product_list': []}
        d_model = {'opt': [], 'v_index': {}, 'c_index': {}, 'ub': [], 'lb': [], 'b_u': [], 'b_l': [], 'integrality': [],
                   'last_c_index': -1}
        for product in group_product_list:
            plan_dict['product_list'].append(product['product_name'])

            d_model = fill_optimization_func_frac(product, d_model)
            for i in range(len(product['transport_list'][0]['cost'])):
                stock = product['stocks'][i]
                d_model['last_c_index'] += 1
                d_model['c_index'][f"c_a_{product['product_name']}_{stock['stock_name']}"] = d_model['last_c_index']
                d_model['b_u'].append(stock['value'])
                d_model['b_l'].append(-np.inf)
            for i in range(len(product['transport_list'][0]['cost'][0])):
                need = product['needs'][i]
                d_model['last_c_index'] += 1
                d_model['c_index'][f"c_b_{product['product_name']}_{need['need_name']}"] = d_model['last_c_index']
                d_model['b_u'].append(need['value'])
                d_model['b_l'].append(need['value'])
        matrix_array = []
        for i in range(len(d_model['c_index'])):
            matrix_array.append([])
            for j in range(len(d_model['opt'])):
                matrix_array[i].append(0)
        for product in group_product_list:
            matrix_array = fill_matrix_array_frac(product, matrix_array, d_model)
        plan_dict['cons_count'] = len(np.array(matrix_array))

        c = np.array(d_model['opt'])

        cons = LinearConstraint(np.array(matrix_array), np.array(d_model['b_l']), np.array(d_model['b_u']))
        inter = np.array(d_model['integrality'])
        bounds = Bounds(np.array(d_model['lb']), np.array(d_model['ub']))

        start_time = time.perf_counter()
        res = milp(c=c, constraints=cons, integrality=inter, bounds=bounds)
        end_time = time.perf_counter()
        plan_dict['working_time'] = end_time - start_time
        plan_dict['v_index'] = d_model['v_index']
        plan_dict['fun'] = res.fun
        plan_dict['node_count'] = res.mip_node_count
        plan_dict['variables'] = res.x
        plan_dict['success'] = res.success
        # plan_dict['unit'] =
        # plan_dict['iter_count'] = res.nit
        plan_list.append(plan_dict)
        # print("res =", res)
        # print("res =", type(res))

    return plan_list


def calc_cons_time_values(plan_list: list[dict]):
    print(plan_list)


def solve_problem_2(data: dict) -> list[dict]:
    # products_list = data['products_list']
    products_priority_dict = {}
    for product in data['products_list']:
        products_priority_dict[product['product_name']] = product['priority']
    # plan_list = data['plan_list']
    for plan in data['plan_list']:
        priority_sum = 0
        for name in plan['products_names']:
            priority_sum += products_priority_dict[name]
        plan['priority'] = priority_sum/len(plan['products_names'])
    # print(data['plan_list'])
    data['plan_list'] = sorted(data['plan_list'], key=lambda x: x['priority'])
    # print('plan_list = ', data['plan_list'])
    plan_list = []
    for plan in data['plan_list']:
        # print(plan['products_names'])
        plan_dict = {'product_list': plan['products_names']}
        d_model = {'opt': [], 'v_index': {}, 'c_index': {}, 'ub': [], 'lb': [], 'b_u': [], 'b_l': [], 'integrality': [],
                   'last_c_index': -1}
        for name in plan['products_names']:
            product = [pr for pr in data['products_list'] if pr['product_name'] == name][0]
            # print('name = ',name, ' product = ', product)
            # ------
            d_model['cons_time_values'] = calc_cons_time_values(plan_list)
            # ------
            d_model = fill_optimization_func_frac(product, d_model)
            for i in range(len(product['transport_list'][0]['cost'])):
                stock = product['stocks'][i]
                d_model['last_c_index'] += 1
                d_model['c_index'][f"c_a_{product['product_name']}_{stock['stock_name']}"] = d_model['last_c_index']
                d_model['b_u'].append(stock['value'])
                d_model['b_l'].append(-np.inf)
            for i in range(len(product['transport_list'][0]['cost'][0])):
                need = product['needs'][i]
                d_model['last_c_index'] += 1
                d_model['c_index'][f"c_b_{product['product_name']}_{need['need_name']}"] = d_model['last_c_index']
                d_model['b_u'].append(need['value'])
                d_model['b_l'].append(need['value'])
        matrix_array = []
        for i in range(len(d_model['c_index'])):
            matrix_array.append([])
            for j in range(len(d_model['opt'])):
                matrix_array[i].append(0)
        for name in plan['products_names']:
            product = [pr for pr in data['products_list'] if pr['product_name'] == name][0]
            matrix_array = fill_matrix_array_frac(product, matrix_array, d_model)
        plan_dict['cons_count'] = len(np.array(matrix_array))

        c = np.array(d_model['opt'])

        cons = LinearConstraint(np.array(matrix_array), np.array(d_model['b_l']), np.array(d_model['b_u']))
        inter = np.array(d_model['integrality'])
        bounds = Bounds(np.array(d_model['lb']), np.array(d_model['ub']))

        start_time = time.perf_counter()
        res = milp(c=c, constraints=cons, integrality=inter, bounds=bounds)
        end_time = time.perf_counter()
        plan_dict['working_time'] = end_time - start_time
        plan_dict['v_index'] = d_model['v_index']
        plan_dict['fun'] = res.fun
        plan_dict['node_count'] = res.mip_node_count
        plan_dict['variables'] = res.x
        plan_dict['success'] = res.success
        plan_list.append(plan_dict)
    return plan_list


def convert_res(plan_list: list[dict], products_list: list[dict]) -> list[dict]:
    for plan in plan_list:
        # print(plan, '\n\n')
        if plan['success']:
            var_values = plan['variables']
            v_index = plan['v_index']
            prev_product_name = ''
            prev_transport_name = ''
            var_list = []
            c_a = {}
            c_b = {}
            c_s = {}
            # cons_dict = {'c_a': [], 'c_b': [], 'c_s': []}
            for i in range(len(var_values)):
                if var_values[i] != 0 and round(var_values[i], 3) != 0:
                    name = [k for k, v in v_index.items() if v == i][0]
                    temp = name.split('_')
                    if temp[0] == 'r':
                        # unit, если ответ дан в рейсах, то 'trips',
                        # иначе (в каких-то единицах измерения тоннах, кг итп) 'other'
                        product_name = temp[1]
                        # print(product_name)
                        transport_name = temp[2]
                        stock_name = temp[3]  # откуда (склад/поставщик)
                        need_name = temp[4]  # куда (МО/потребитель)
                        if product_name != prev_product_name:
                            product = [pr for pr in products_list if pr['product_name'] == product_name][0]
                            prev_product_name = product['product_name']
                        if transport_name != prev_transport_name:
                            transport = \
                            [tr for tr in product['transport_list'] if tr['transport_name'] == transport_name][0]
                            gk = transport['gk']
                            time = transport['time']
                            prev_transport_name = transport['transport_name']
                        for index, value in enumerate(product['stocks']):
                            if value['stock_name'] == stock_name:
                                stock_index = index
                                break
                        for index, value in enumerate(product['needs']):
                            if value['need_name'] == need_name:
                                need_index = index
                                break
                        # stock_index = [index for index, value in enumerate(product['stocks']) if value['stock_name'] == stock_name][0]
                        var_value = var_values[i]
                        gk_var_value = gk[stock_index][need_index] * var_value
                        if c_a.get(f'c_a_{product_name}_{stock_name}') is None:
                            c_a[f'c_a_{product_name}_{stock_name}'] = gk_var_value
                        else:
                            c_a[f'c_a_{product_name}_{stock_name}'] += gk_var_value

                        if c_b.get(f'c_b_{product_name}_{need_name}') is None:
                            c_b[f'c_b_{product_name}_{need_name}'] = gk_var_value
                        else:
                            c_b[f'c_b_{product_name}_{need_name}'] += gk_var_value

                        group = product["group"]
                        if c_s.get(f'c_s_{group}_{transport_name}_{stock_name}') == None:
                            c_s[f'c_s_{group}_{transport_name}_{stock_name}'] = time[stock_index][need_index] * ceil(var_value)
                        else:
                            c_s[f'c_s_{group}_{transport_name}_{stock_name}'] += time[stock_index][need_index] * ceil(var_value)

                        if product['unit'] == 'trips':
                            value_trips = var_value
                        else:
                            value_trips = gk_var_value
                        var_list.append({'value': value_trips, 'unit': product['unit'],
                                         'product_name': product_name, 'transport': transport_name,
                                         'stock_name': stock_name, 'need_name': need_name})
            # plan['var_list'] = var_list
            plan['trips'] = var_list
            plan['c_a'] = c_a
            plan['c_b'] = c_b
            plan['c_s'] = c_s
        else:
            plan['trips'] = None
        del plan['v_index']
        del plan['variables']
    return plan_list


# script_name, first = argv
# print("script_name: ", script_name)
# print("first variable type: ", type(first), ' variable: ', first)
# print("model = ", model)

# Открываем файл для чтения
# path_to_data = "E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\module_data\\module_input_data_5x12.json"
# path_to_data = "E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\module_data\\module_input_data_2x2.json"
path_to_data = "E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\module_data\\module_input_data_2x2_small_s.json"

with open(path_to_data, 'r', encoding='utf-8') as file:
    # Загружаем содержимое файла в переменную
    data = json.load(file)

# print(data)
# plan_list = solve_problem(data)
plan_list = solve_problem_2(data)
# print(plan_list)
res = convert_res(plan_list, data['products_list'])
# print(res[1])
# print(res[1]['c_s'])
