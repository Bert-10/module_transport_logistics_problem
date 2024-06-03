import openpyxl
# from scipy.optimize import linprog
import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds
from math import ceil
import time
import re
from functools import partial
from sys import argv
import json


def fill_optimization_func(product, d_model: dict, c_s, fill_func) -> dict:
    check_time = False
    product_name = product['product_name']
    product_time = product['time']
    product_group = product["group"]
    for transport in product['transport_list']:
        transport_name = transport['transport_name']
        for row in range(len(transport['cost'])):
            stock_name = product['stocks'][row]['stock_name']
            for column in range(len(transport['cost'][0])):
                if transport['cost'][row][column] != 'nan':
                    d_model = fill_func(d_model, transport, product_name, transport_name, stock_name,
                                        product['needs'][column]['need_name'], row, column)
                    check_time = True
            if check_time:
                check_time = False
                if f'c_s_{product_group}_{transport_name}_{stock_name}' not in d_model['c_index'].keys():
                    d_model['last_c_index'] += 1
                    d_model['c_index'][f'c_s_{product_group}_{transport_name}_{stock_name}'] = d_model[
                        'last_c_index']
                    if f'c_s_{product_group}_{transport_name}_{stock_name}' in c_s:
                        d_model['b_u'].append(c_s[f'c_s_{product_group}_{transport_name}_{stock_name}'])
                    else:
                        d_model['b_u'].append(product_time)
                    d_model['b_l'].append(-np.inf)
    # d_model['last_c_index'] = last_c_index
    return d_model


def fill_optimization_func_cost_frac(d_model: dict, transport: dict, product_name: str, transport_name: str,
                                     stock_name: str,
                                     need_name: str, row: int, column: int) -> dict:
    last_c_index = d_model['last_c_index']
    b_u = d_model['b_u']
    b_l = d_model['b_l']
    opt = d_model['opt']
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    lb = d_model['lb']
    ub = d_model['ub']
    integrality = d_model['integrality']
    cost = transport['cost']
    gk = transport['gk']
    calc_trips = transport['calc_trips']
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
    d_model['last_c_index'] = last_c_index

    return d_model


def fill_optimization_func_cost_whole(d_model: dict, transport: dict, product_name: str, transport_name: str,
                                      stock_name: str,
                                      need_name: str, row: int, column: int) -> dict:
    last_c_index = d_model['last_c_index']
    b_u = d_model['b_u']
    b_l = d_model['b_l']
    opt = d_model['opt']
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    lb = d_model['lb']
    ub = d_model['ub']
    integrality = d_model['integrality']
    cost = transport['cost']
    gk = transport['gk']
    calc_trips = transport['calc_trips']
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
    ub.append(gk[row][column] - 1)
    lb.append(0)
    integrality.append(0)

    last_c_index += 1
    c_index[f'c_z_r_{product_name}_{transport_name}_{stock_name}_{need_name}'] = last_c_index
    b_u.append(0)
    b_l.append(-np.inf)
    d_model['last_c_index'] = last_c_index

    return d_model


def fill_optimization_func_time_frac(d_model: dict, transport: dict, product_name: str, transport_name: str,
                                     stock_name: str,
                                     need_name: str, row: int, column: int) -> dict:
    last_c_index = d_model['last_c_index']
    b_u = d_model['b_u']
    b_l = d_model['b_l']
    opt = d_model['opt']
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    lb = d_model['lb']
    ub = d_model['ub']
    integrality = d_model['integrality']
    # cost = transport['cost']
    time = transport['time']
    # calc_trips = transport['calc_trips']
    v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"] = len(opt)
    opt.append(time[row][column])
    ub.append(np.inf)
    lb.append(0)
    integrality.append(0)
    v_index[f'w_{product_name}_{transport_name}_{stock_name}_{need_name}'] = len(opt)
    opt.append(1)
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
    d_model['last_c_index'] = last_c_index

    return d_model


def fill_optimization_func_time_whole(d_model: dict, transport: dict, product_name: str, transport_name: str,
                                      stock_name: str,
                                      need_name: str, row: int, column: int) -> dict:
    last_c_index = d_model['last_c_index']
    b_u = d_model['b_u']
    b_l = d_model['b_l']
    opt = d_model['opt']
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    lb = d_model['lb']
    ub = d_model['ub']
    integrality = d_model['integrality']
    # cost = transport['cost']
    time = transport['time']
    gk = transport['gk']
    # calc_trips = transport['calc_trips']
    v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"] = len(opt)
    opt.append(time[row][column])
    ub.append(np.inf)
    lb.append(0)
    integrality.append(1)
    v_index[f'z_{product_name}_{transport_name}_{stock_name}_{need_name}'] = len(opt)
    opt.append(0)
    ub.append(gk[row][column] - 1)
    lb.append(0)
    integrality.append(0)

    last_c_index += 1
    c_index[f'c_z_r_{product_name}_{transport_name}_{stock_name}_{need_name}'] = last_c_index
    b_u.append(0)
    b_l.append(-np.inf)
    d_model['last_c_index'] = last_c_index

    return d_model


def fill_matrix_array_cost(product: dict, matrix_array: list[list[float]], d_model: dict, fill_func) -> list[
    list[float]]:
    # print('product_GROUP = =', product['group'])
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    product_name = product['product_name']
    product_group = product["group"]
    for transport in product['transport_list']:
        transport_name = transport['transport_name']
        gk = transport['gk']
        time = transport['time']
        for row in range(len(transport['cost'])):
            stock_name = product['stocks'][row]['stock_name']
            for column in range(len(transport['cost'][0])):
                if transport['cost'][row][column] != 'nan':
                    matrix_array = fill_func(matrix_array, v_index, c_index, product_name, transport_name, gk, time,
                                             stock_name, product['needs'][column]['need_name'], product_group, row,
                                             column)
    return matrix_array


def fill_matrix_array_cost_frac(matrix_array, v_index, c_index, product_name, transport_name, gk, time, stock_name,
                                need_name, group, row, column, big_number=1000) -> list[list[float]]:
    # print('product_name = ', matrix_array, ' stock_name = ', stock_name)
    i = c_index[f'c_a_{product_name}_{stock_name}']
    j_r = v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"]
    matrix_array[i][j_r] = gk[row][column]

    i = c_index[f'c_b_{product_name}_{need_name}']
    matrix_array[i][j_r] = gk[row][column]

    i = c_index[f'c_s_{group}_{transport_name}_{stock_name}']
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


def fill_matrix_array_cost_whole(matrix_array, v_index, c_index, product_name, transport_name, gk, time, stock_name,
                                 need_name, group, row, column) -> list[list[float]]:
    i = c_index[f'c_a_{product_name}_{stock_name}']
    j_r = v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"]
    j_z = v_index[f'z_{product_name}_{transport_name}_{stock_name}_{need_name}']
    matrix_array[i][j_r] = gk[row][column]
    matrix_array[i][j_z] = -1

    i = c_index[f'c_b_{product_name}_{need_name}']
    matrix_array[i][j_r] = gk[row][column]
    matrix_array[i][j_z] = -1

    i = c_index[f'c_s_{group}_{transport_name}_{stock_name}']
    matrix_array[i][j_r] = time[row][column]

    i = c_index[f'c_z_r_{product_name}_{transport_name}_{stock_name}_{need_name}']
    matrix_array[i][j_r] = -gk[row][column]
    matrix_array[i][j_z] = 1

    return matrix_array


def fill_matrix_array_time(product: dict, matrix_array: list[list[float]], d_model: dict, fill_func) -> list[
    list[float]]:
    # print('product_GROUP = =', product['group'])
    v_index = d_model['v_index']
    c_index = d_model['c_index']
    product_name = product['product_name']
    product_group = product["group"]
    for transport in product['transport_list']:
        transport_name = transport['transport_name']
        gk = transport['gk']
        time = transport['time']
        cost = transport['cost']
        calc_trips = transport['calc_trips']
        for row in range(len(transport['cost'])):
            stock_name = product['stocks'][row]['stock_name']
            for column in range(len(transport['cost'][0])):
                if transport['cost'][row][column] != 'nan':
                    matrix_array = fill_func(matrix_array, v_index, c_index, product_name, transport_name, cost, gk,
                                             time, calc_trips,
                                             stock_name, product['needs'][column]['need_name'], product_group, row,
                                             column)
    return matrix_array


def fill_matrix_array_time_frac(matrix_array, v_index, c_index, product_name, transport_name, cost, gk, time,
                                calc_trips, stock_name,
                                need_name, group, row, column, big_number=1000) -> list[list[float]]:
    # print('product_name = ', matrix_array, ' stock_name = ', stock_name)
    i = c_index[f'c_a_{product_name}_{stock_name}']
    j_r = v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"]
    matrix_array[i][j_r] = gk[row][column]

    i = c_index[f'c_b_{product_name}_{need_name}']
    matrix_array[i][j_r] = gk[row][column]

    i = c_index[f'c_s_{group}_{transport_name}_{stock_name}']
    j_w = v_index[f'w_{product_name}_{transport_name}_{stock_name}_{need_name}']
    matrix_array[i][j_r] = time[row][column]
    matrix_array[i][j_w] = time[row][column]

    i = c_index[f'c_gw_r_{product_name}_{transport_name}_{stock_name}_{need_name}']
    matrix_array[i][j_r] = -1
    matrix_array[i][j_w] = big_number

    i = c_index[f'c_w_r_{product_name}_{transport_name}_{stock_name}_{need_name}']
    matrix_array[i][j_r] = -1
    matrix_array[i][j_w] = 1

    i = c_index[f'c_cost']
    if calc_trips[row][column] == 1:
        matrix_array[i][j_r] = cost[row][column]
    else:
        matrix_array[i][j_r] = cost[row][column] * gk[row][column]
    return matrix_array


def fill_matrix_array_time_whole(matrix_array, v_index, c_index, product_name, transport_name, cost, gk, time,
                                 calc_trips, stock_name,
                                 need_name, group, row, column) -> list[list[float]]:
    i = c_index[f'c_a_{product_name}_{stock_name}']
    j_r = v_index[f"r_{product_name}_{transport_name}_{stock_name}_{need_name}"]
    j_z = v_index[f'z_{product_name}_{transport_name}_{stock_name}_{need_name}']
    matrix_array[i][j_r] = gk[row][column]
    matrix_array[i][j_z] = -1

    i = c_index[f'c_b_{product_name}_{need_name}']
    matrix_array[i][j_r] = gk[row][column]
    matrix_array[i][j_z] = -1

    i = c_index[f'c_s_{group}_{transport_name}_{stock_name}']
    matrix_array[i][j_r] = time[row][column]

    i = c_index[f'c_z_r_{product_name}_{transport_name}_{stock_name}_{need_name}']
    matrix_array[i][j_r] = -gk[row][column]
    matrix_array[i][j_z] = 1

    i = c_index[f'c_cost']
    if calc_trips[row][column] == 1:
        matrix_array[i][j_r] = cost[row][column]
    else:
        matrix_array[i][j_r] = cost[row][column] * gk[row][column]
        matrix_array[i][j_z] = -cost[row][column]

    return matrix_array


def find_el_for_calc_cons(temp, prev_product_name, prev_transport_name, products_list):
    product_name = temp[1]
    transport_name = temp[2]
    stock_name = temp[3]  # откуда (склад/поставщик)
    need_name = temp[4]  # куда (МО/потребитель)
    if product_name != prev_product_name:
        product = [pr for pr in products_list if pr['product_name'] == product_name][0]
    if transport_name != prev_transport_name:
        transport = \
            [tr for tr in product['transport_list'] if tr['transport_name'] == transport_name][0]
    for index, value in enumerate(product['stocks']):
        if value['stock_name'] == stock_name:
            stock_index = index
            break
    for index, value in enumerate(product['needs']):
        if value['need_name'] == need_name:
            need_index = index
            break

    return {'product': product, 'transport': transport, 'stock_index': stock_index, 'need_index': need_index}


def calc_cons_cost_values(c_cost, plan_list: list[dict], products_list):
    if plan_list != []:
        plan = plan_list[len(plan_list) - 1]
        if plan['success'] and plan['opt_func'] == 'time':
            var_values = plan['variables']
            v_index = plan['v_index']
            prev_product_name = ''
            prev_transport_name = ''
            for i in range(len(var_values)):
                if var_values[i] != 0 and round(var_values[i], 3) != 0:
                    name = [k for k, v in v_index.items() if v == i][0]
                    temp = name.split('_')
                    if temp[0] == 'r':
                        res = find_el_for_calc_cons(temp, prev_product_name, prev_transport_name, products_list)
                        stock_index = res['stock_index']
                        need_index = res['need_index']

                        var_value = var_values[i]
                        if res['transport']['calc_trips'][stock_index][need_index] == 1:
                            c_cost -= res['transport']['cost'][stock_index][need_index] * ceil(var_value)
                        else:
                            c_cost -= res['transport']['cost'][stock_index][need_index] * \
                                      res['transport']['gk'][stock_index][need_index] * var_value
                    elif temp[0] == 'z':
                        res = find_el_for_calc_cons(temp, prev_product_name, prev_transport_name, products_list)
                        stock_index = res['stock_index']
                        need_index = res['need_index']

                        var_value = var_values[i]
                        if res['transport']['calc_trips'][stock_index][need_index] != 1:
                            c_cost -= res['transport']['cost'][stock_index][need_index] * var_value
    return c_cost


def calc_cons_time_values(c_s, plan_list: list[dict], products_list):
    if plan_list != []:
        plan = plan_list[len(plan_list) - 1]
        if plan['success']:
            var_values = plan['variables']
            v_index = plan['v_index']
            prev_product_name = ''
            prev_transport_name = ''
            for i in range(len(var_values)):
                if var_values[i] != 0 and round(var_values[i], 3) != 0:
                    name = [k for k, v in v_index.items() if v == i][0]
                    temp = name.split('_')
                    if temp[0] == 'r':
                        res = find_el_for_calc_cons(temp, prev_product_name, prev_transport_name, products_list)

                        var_value = var_values[i]
                        cons_name = f"c_s_{res['product']['group']}_{res['transport']['transport_name']}_{temp[3]}"
                        if c_s.get(cons_name) is None:
                            c_s[cons_name] = res['product']['time'] - res['transport']['time'][res['stock_index']][
                                res['need_index']] * ceil(var_value)
                        else:
                            c_s[cons_name] -= res['transport']['time'][res['stock_index']][res['need_index']] * ceil(
                                var_value)
                        if c_s[cons_name] < 0:
                            c_s[cons_name] = 0
    return c_s


def is_number(s):
    s = str(s)
    if re.match("^\d+?\.\d+?$", s) is None:
        return s.isdigit()
    return True


def solve_problem(data: dict) -> list[dict]:
    # products_priority_dict = {}
    # for product in data['products_list']:
    #     products_priority_dict[product['product_name']] = product['priority']
    # for plan in data['plan_list']:
    #     priority_sum = 0
    #     for name in plan['products_names']:
    #         priority_sum += products_priority_dict[name]
    #     plan['priority'] = priority_sum / len(plan['products_names'])

    data['plan_list'] = sorted(data['plan_list'], key=lambda x: x['priority'])
    # print('plan_list = ', data['plan_list'])
    plan_list = []
    big_number = 1000
    # big_number = np.inf
    cons_time_values = {}
    c_cost = 'nan'
    if is_number(data['max_cost']):
        c_cost = float(data['max_cost'])
    for plan in data['plan_list']:
        # print(plan)
        plan_dict = {'product_list': plan['products_names']}
        d_model = {'opt': [], 'v_index': {}, 'c_index': {}, 'ub': [], 'lb': [], 'b_u': [], 'b_l': [], 'integrality': [],
                   'last_c_index': -1}

        if plan['opt_func'] == 'cost':
            if plan['model'] == 'fractional':
                # print('cost -> fractional')
                def_optimization_func = fill_optimization_func_cost_frac
                fill_matrix_array = partial(fill_matrix_array_cost,
                                            fill_func=partial(fill_matrix_array_cost_frac, big_number=big_number))
                # def_matrix_array = partial(fill_matrix_array_cost_frac, big_number=big_number)
            else:
                # print('cost -> whole')
                def_optimization_func = fill_optimization_func_cost_whole
                fill_matrix_array = partial(fill_matrix_array_cost, fill_func=fill_matrix_array_cost_whole)
                # def_matrix_array = fill_matrix_array_cost_whole
        else:
            # print(data['max_cost'])
            if c_cost != 'nan':
                # print('BOOOOOOOO')
                d_model['last_c_index'] += 1
                d_model['c_index']['c_cost'] = d_model['last_c_index']
                # d_model['b_u'].append(float(data['max_cost']))
                d_model['b_u'].append(calc_cons_cost_values(c_cost, plan_list, data['products_list']))
                d_model['b_l'].append(-np.inf)
                if plan['model'] == 'fractional':
                    def_optimization_func = fill_optimization_func_time_frac
                    fill_matrix_array = partial(fill_matrix_array_time,
                                                fill_func=partial(fill_matrix_array_time_frac, big_number=big_number))
                else:
                    def_optimization_func = fill_optimization_func_time_whole
                    fill_matrix_array = partial(fill_matrix_array_time, fill_func=fill_matrix_array_time_whole)
            else:
                # print('HEREEEEEEEEEEEEEEEE')
                if plan['model'] == 'fractional':
                    def_optimization_func = fill_optimization_func_time_frac
                    fill_matrix_array = partial(fill_matrix_array_cost,
                                                fill_func=partial(fill_matrix_array_cost_frac, big_number=big_number))
                else:
                    def_optimization_func = fill_optimization_func_time_whole
                    fill_matrix_array = partial(fill_matrix_array_cost, fill_func=fill_matrix_array_cost_whole)

        cons_time_values = calc_cons_time_values(cons_time_values, plan_list, data['products_list'])
        for name in plan['products_names']:
            product = [pr for pr in data['products_list'] if pr['product_name'] == name][0]
            # print('name = ',name, ' product = ', product)
            d_model = fill_optimization_func(product, d_model, cons_time_values, def_optimization_func)
            for i in range(len(product['transport_list'][0]['cost'])):
                stock = product['stocks'][i]
                # print(d_model['last_c_index'], '  ', type(d_model['last_c_index']))
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
            matrix_array = fill_matrix_array(product, matrix_array, d_model)
            # matrix_array = fill_matrix_array(product, matrix_array, d_model, def_matrix_array)
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
        plan_dict['model'] = plan['model']
        plan_dict['opt_func'] = plan['opt_func']
        plan_list.append(plan_dict)
    return plan_list


def convert_res(plan_list: list[dict], products_list: list[dict]) -> dict[str, list[dict]]:
    decimal_places = 3
    for plan in plan_list:
        if plan['success']:
            var_values = plan['variables']
            v_index = plan['v_index']
            prev_product_name = ''
            prev_transport_name = ''
            var_list = []
            c_a = {}
            c_b = {}
            c_s = {}
            c_cost = 0
            # if plan['opt_func'] == 'time':
            #     c_cost_calc_def =
            # else:
            #     c_cost_calc_def = 'pass'
            for i in range(len(var_values)):
                if var_values[i] != 0 and round(var_values[i], decimal_places) != 0:
                    name = [k for k, v in v_index.items() if v == i][0]
                    temp = name.split('_')
                    if temp[0] == 'r':
                        product_name = temp[1]
                        transport_name = temp[2]
                        stock_name = temp[3]  # откуда (склад/поставщик)
                        need_name = temp[4]  # куда (МО/потребитель)
                        res = find_el_for_calc_cons(temp, prev_product_name, prev_transport_name, products_list)

                        stock_index = res['stock_index']
                        need_index = res['need_index']
                        transport = res['transport']
                        var_value = var_values[i]

                        gk_var_value = transport['gk'][stock_index][need_index] * var_value
                        if c_a.get(f'c_a_{product_name}_{stock_name}') is None:
                            c_a[f'c_a_{product_name}_{stock_name}'] = gk_var_value
                        else:
                            c_a[f'c_a_{product_name}_{stock_name}'] += gk_var_value

                        if c_b.get(f'c_b_{product_name}_{need_name}') is None:
                            c_b[f'c_b_{product_name}_{need_name}'] = gk_var_value
                        else:
                            c_b[f'c_b_{product_name}_{need_name}'] += gk_var_value

                        group = res['product']["group"]
                        if c_s.get(f'c_s_{group}_{transport_name}_{stock_name}') is None:
                            c_s[f'c_s_{group}_{transport_name}_{stock_name}'] = transport['time'][stock_index][
                                                                                    need_index] * ceil(
                                var_value)
                        else:
                            c_s[f'c_s_{group}_{transport_name}_{stock_name}'] += transport['time'][stock_index][
                                                                                     need_index] * ceil(
                                var_value)

                        unit = res['product']['unit']
                        if unit == 'trips':
                            value_trips = var_value
                        else:
                            value_trips = gk_var_value
                        var_list.append({'value': value_trips, 'unit': unit,
                                         'product_name': product_name, 'transport': transport_name,
                                         'stock_name': stock_name, 'need_name': need_name})
                        if plan['opt_func'] == 'time':
                            if transport['calc_trips'][stock_index][need_index] == 1:
                                c_cost += transport['cost'][stock_index][need_index] * ceil(var_value)
                            else:
                                c_cost += transport['cost'][stock_index][need_index] * gk_var_value
                    elif temp[0] == 'z':
                        product_name = temp[1]
                        stock_name = temp[3]  # откуда (склад/поставщик)
                        need_name = temp[4]  # куда (МО/потребитель)
                        var_value = var_values[i]
                        if c_a.get(f'c_a_{product_name}_{stock_name}') is None:
                            c_a[f'c_a_{product_name}_{stock_name}'] = -var_value
                        else:
                            c_a[f'c_a_{product_name}_{stock_name}'] -= var_value

                        if c_b.get(f'c_b_{product_name}_{need_name}') is None:
                            c_b[f'c_b_{product_name}_{need_name}'] = -var_value
                        else:
                            c_b[f'c_b_{product_name}_{need_name}'] -= var_value

                        if plan['opt_func'] == 'time':
                            res = find_el_for_calc_cons(temp, prev_product_name, prev_transport_name, products_list)
                            stock_index = res['stock_index']
                            need_index = res['need_index']
                            transport = res['transport']
                            if transport['calc_trips'][stock_index][need_index] != 1:
                                c_cost -= transport['cost'][stock_index][need_index] * var_value

            plan['variables_count'] = len(var_values)
            plan['trips'] = var_list
            plan['working_time'] = round(plan['working_time'], decimal_places)
            plan['fun'] = round(plan['fun'], decimal_places)
            plan['c_a'] = c_a
            plan['c_b'] = c_b
            plan['c_s'] = c_s
            if c_cost != 0:
                plan['c_cost'] = c_cost
        else:
            plan['trips'] = None
        del plan['v_index']
        del plan['variables']

    for plan in plan_list:
        if plan['success']:
            trips = plan['trips']
            for el in trips:
                el['value'] = float(round(el['value'], decimal_places))
            c_a = []
            c_b = []
            c_s = []
            for name, value in plan['c_a'].items():
                splited_name = name.split('_')
                c_a.append({'value': float(round(value, decimal_places)),
                            'product_name': splited_name[2],
                            'stock_name': splited_name[3]})
            for name, value in plan['c_b'].items():
                splited_name = name.split('_')
                c_b.append({'value': float(round(value, decimal_places)),
                            'product_name': splited_name[2],
                            'need_name': splited_name[3]})
            for name, value in plan['c_s'].items():
                splited_name = name.split('_')
                c_s.append({'value': float(round(value, decimal_places)),
                            'group': splited_name[2],
                            'transport': splited_name[3],
                            'stock_name': splited_name[4]})
            plan['c_a'] = c_a
            plan['c_b'] = c_b
            plan['c_s'] = c_s
    # print("plan_list = ", plan_list)
    return {"plan_list": plan_list}


script_name, path_to_input_data, path_to_save_data = argv
# print("script_name: ", script_name)
# print("first variable type: ", type(first), ' variable: ', first)
# print("model = ", model)

# Открываем файл для чтения
# path_to_input_data = "E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\module_data\\module_input_data_2x2.json"
# path_to_save_data = "E:\\Папка рабочего стола\\VScodeProjects\\vkr_js\\module_output_data\\module_output_data.json"

with open(path_to_input_data, 'r', encoding='utf-8') as file:
    # Загружаем содержимое файла в переменную
    data = json.load(file)


plan_list = solve_problem(data)
res = convert_res(plan_list, data['products_list'])
# for k, v in res['plan_list'][0].items():
#     print(f'{k} = ', v)
# print(res)

with open(path_to_save_data, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
# print('res[0] = ', res[0])
# print('res[1] = ', res[1])
# print('res[1]["c_s"] = ', res[1]['c_s'])
# print('res[1]["c_a"] = ', res[1]['c_a'])
# print('res[1]["c_b"] = ', res[1]['c_b'])
# print('res[1]["c_cost"] = ', res[1]['c_cost'])
# print('res[0]["c_cost"] = ', res[0]['c_cost'])
