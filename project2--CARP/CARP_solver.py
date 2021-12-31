# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import copy
import math
import random
import sys
import time
import numpy as np
import multiprocessing as mp

max_time = 0
start_time = time.time()
distance = []
capacity = 0
depot = 0
dem = []
seed = 0
table_all = []


def arg_analysis():
    parser = argparse.ArgumentParser(description="deal with args")
    parser.add_argument("file_name")
    parser.add_argument("-t", type=int, default=60)
    parser.add_argument("-s", type=int, default=time.time())
    args = parser.parse_args()

    return args.file_name, args.t, args.s


def read_file(file_name):
    with open(file_name, 'r+') as file:
        name = file.readline().split()[2]
        vertices = int(file.readline().split()[2])
        depot = int(file.readline().split()[2])
        required_edges = int(file.readline().split()[3])
        non_required_edges = int(file.readline().split()[3])
        vehicles = int(file.readline().split()[2])
        capacity = int(file.readline().split()[2])
        total = int(file.readline().split()[6])
        file.readline()
        graph = 99999 * np.ones((vertices + 1, vertices + 1),
                                dtype=np.int32)
        demands = []
        while True:
            now = file.readline()
            if now == "END":
                break
            nodes = now.split()
            node1 = int(nodes[0])
            node2 = int(nodes[1])
            cost = int(nodes[2])
            demand = int(nodes[3])
            graph[node1][node2] = cost
            graph[node2][node1] = cost
            if demand != 0:
                demands.append((node1, node2, cost, demand))
                demands.append((node2, node1, cost, demand))

        distance = floyd(graph)
        demands = np.array(demands)
        demands = demands.tolist()

    return depot, capacity, demands, distance


def floyd(graph):
    n = len(graph)
    for k in range(1, n):
        graph[k][k] = 0
        for i in range(1, n):
            for j in range(1, n):
                if graph[i][j] > graph[i][k] + graph[k][j]:
                    graph[i][j] = graph[i][k] + graph[k][j]
    return graph


def get_init(demands, weight):
    # copy_demand = copy.deepcopy(demands)
    # a1, b1, c1 = path_scanning(copy_demand, 0)
    population = []
    do_time = 0
    # if max_time > 120:
    if weight[0] > 0.05:
        do_time = 0
    else:
        do_time = 0.5 * max_time
    for i in [1, 2]:
        copy_demand = copy.deepcopy(demands)
        a2, b2, c2 = path_scanning(copy_demand, i)
        population.append((a2, b2, c2))

    # cho = random.randint(0, 8)
    cho = 0
    global table
    # table.add(get_hash(b1))
    # table.add(get_hash(b2))
    # cho = 0
    # cho=0
    if cho < 0:
        cho = 0

    while time.time() - start_time < do_time:
        a = 1
        copy_demand = copy.deepcopy(demands)
        a, b, c = path_scanning_random(copy_demand)
        population.append((a, b, c))
        # table.add(get_hash(b))
    list = sorted(population, key=lambda p: p[0])
    # if random.random()<0.5:
    #     list.insert(0,population[0])
    if cho >= len(list) - 1:
        cho = 0
    # return population[0]
    two_opt(population[0][1], population[0][2])
    two_opt(list[0][1], list[0][2])
    two_opt(list[cho][1], list[cho][2])

    if weight[0] > 0.05:
        return population[0]
    if list[cho][0] > 1.05 * list[0][0]:
        return list[0]
    else:
        return list[cho]


def get_hash(routes):
    h = 0
    for route in routes:
        for task in route:
            h = (31 * h + task[0]) % 1000000007
    return h


def path_scanning(demands, ch):
    ans = []
    total_cost = 0
    routes = []
    remain = []
    while len(demands) != 0:
        ans.append(0)
        route = []
        route_full_info = []
        load = 0
        o = depot
        route_cost = 0
        while True:
            candidates = []
            for d in demands:
                if d[3] + load <= capacity:
                    candidates.append(d)
            if len(candidates) == 0:
                break
            find_min = False
            choose = []
            # if load < capacity / 2:
            #     m = -9999999
            # else:
            m = 9999999

            find_min = True

            for task in candidates:
                cost = distance[o][task[0]] + task[2]
                if ch == 1:
                    cost0 = distance[o][task[0]]
                else:
                    cost0 = distance[o][task[0]] / task[3]
                if find_min and cost0 < m or (not find_min and cost > m):
                    m = cost0
                    choose = task
            route.append((choose[0], choose[1]))
            route_full_info.append(choose)
            demands.remove(choose)
            demands.remove([choose[1], choose[0], choose[2], choose[3]])
            route_cost += distance[o][choose[0]] + choose[2]
            # print(f"{o} to {choose[0]} , cost {distance[o][choose[0]]}")
            # print(f"{choose[0]} to {choose[1]} , cost {choose[2]}")
            load += choose[3]
            o = choose[1]
        # total_cost += distance[o][depot]
        route_cost += distance[o][depot]
        total_cost += route_cost
        # print(f"{o} to {depot} , cost {distance[o][depot]}")
        ans.extend(route)
        remain.append([capacity - load, route_cost])
        # if o != depot:
        #     ans.append((o, depot))
        ans.append(0)
        routes.append(route_full_info)

    # total_cost = flip(routes, total_cost)
    return total_cost, routes, remain


def path_scanning_random(demands):
    ans = []
    total_cost = 0
    routes = []
    remain = []
    while len(demands) != 0:
        ans.append(0)
        route = []
        route_full_info = []
        load = 0
        o = depot
        route_cost = 0
        while True:
            candidates = []
            for d in demands:
                if d[3] + load <= capacity:
                    candidates.append(d)
            if len(candidates) == 0:
                break
            find_min = False
            choose = []
            # if load < capacity / 2:
            #     m = -9999999
            # else:
            m = 9999999

            choose_list = []
            find_min = True
            for task in candidates:
                cost = distance[o][task[0]]
                if cost == m:
                    choose_list.append(task)
                if find_min and cost < m or (not find_min and cost > m):
                    m = cost
                    choose_list = [task]
                # elif float(cost) < float(1.01 * m) and random.random() < 0.4:
                #     choose_list.append(task)
                # elif float(cost) < float(1.03 * m) and random.random() < 0.2:
                #     choose_list.append(task)
                # elif float(cost) < float(1.08 * m) and random.random() < 0.1:
                #     choose_list.append(task)
                # elif float(cost) < float(1.1 * m) and random.random() < 0.05:
                #     choose_list.append(task)
            r = random.randint(0, len(choose_list) - 1)
            choose = choose_list[r]
            route.append((choose[0], choose[1]))
            route_full_info.append(choose)
            demands.remove(choose)
            demands.remove([choose[1], choose[0], choose[2], choose[3]])
            route_cost += distance[o][choose[0]] + choose[2]
            # print(f"{o} to {choose[0]} , cost {distance[o][choose[0]]}")
            # print(f"{choose[0]} to {choose[1]} , cost {choose[2]}")
            load += choose[3]
            o = choose[1]
        # total_cost += distance[o][depot]
        route_cost += distance[o][depot]
        total_cost += route_cost
        # print(f"{o} to {depot} , cost {distance[o][depot]}")
        ans.extend(route)
        remain.append([capacity - load, route_cost])
        # if o != depot:
        #     ans.append((o, depot))
        ans.append(0)
        routes.append(route_full_info)
    # total_cost = flip(routes, total_cost)
    return total_cost, routes, remain


def flip1(ans, cost):
    for i in range(len(ans)):
        if ans[i] != 0:
            if i == 0:
                start = depot
            elif ans[i - 1] == 0:
                start = depot
            else:
                start = ans[i - 1][1]
            if i == len(ans) - 1:
                end = depot
            elif ans[i + 1] == 0:
                end = depot
            else:
                end = ans[i + 1][0]
            det = -distance[start][ans[i][0]] - distance[ans[i][1]][end] + distance[end][ans[i][0]] + \
                  distance[ans[i][1]][start]
            if det < 0:
                cost += det
                ans[i] = ans[i][1], ans[i][0]
    return cost


def flip(routes, cost, remain):
    for j in range(len(routes)):
        route = routes[j]
        for i in range(len(route)):
            if i == 0:
                start = depot
            else:
                start = route[i - 1][1]
            if i == len(route) - 1:
                end = depot
            else:
                end = route[i + 1][0]
            det = -distance[start][route[i][0]] - distance[route[i][1]][end] + distance[end][route[i][0]] + \
                  distance[route[i][1]][start]
            if det < 0:
                cost += det
                route[i] = [route[i][1], route[i][0], route[i][2], route[i][3]]
                remain[j][1] += det
    return cost


def recombine(routes, remain):
    r1, r2 = 0, 0
    while True:
        r1 = random.randint(0, len(routes) - 1)
        r2 = random.randint(0, len(routes) - 1)
        if r1 != r2:
            break
    route1 = routes[r1]
    route2 = routes[r2]

    count = 0
    while True:
        r3 = random.randint(0, len(route1) - 1)
        r4 = random.randint(0, len(route2) - 1)
        new_route1 = route1[:r3] + route2[r4:]
        new_route2 = route2[:r4] + route1[r3:]
        new_demand1 = np.sum(new_route1, axis=0)[3]
        new_demand2 = np.sum(new_route2, axis=0)[3]
        count += 1
        if (new_demand1 <= capacity and new_demand2 <= capacity) or count > len(route1) + len(route2):
            break

    if count > len(new_route1) + len(new_route2):
        return 0
    if r3 == 0:
        start1 = depot
    else:
        start1 = route1[r3 - 1][1]
    if r3 == len(route1) - 1:
        end1 = depot
    else:
        end1 = route1[r3 + 1][0]

    if r3 == 0:
        start1 = depot
    else:
        start1 = route1[r3 - 1][1]
    if r3 == len(route1) - 1:
        end1 = depot
    else:
        end1 = route1[r3 + 1][0]
    new_cost1 = cal_new_cost(new_route1)
    new_cost2 = cal_new_cost(new_route2)
    old_cost1 = remain[r1][1]
    old_cost2 = remain[r2][1]
    routes[r1] = new_route1
    routes[r2] = new_route2
    remain[r1][1] = new_cost1
    remain[r2][1] = new_cost2
    remain[r1][0] = capacity - new_demand1
    remain[r2][0] = capacity - new_demand2
    # print(new_cost1 + new_cost2 - old_cost1 - old_cost2)
    return new_cost1 + new_cost2 - old_cost1 - old_cost2


def simulated_annealing(routes, cost, remain, weight, see):
    global table
    T = 10000
    cool_rate = 0.001
    best_routes = routes
    best_cost = cost
    best_remain = remain
    times = 0
    repeat = 0
    # print(f"----{cost}")
    while time.time() - start_time < max_time - 1:
        times += 1

        last_cost = cost
        copy_routes = copy.deepcopy(routes)
        copy_remain = copy.deepcopy(remain)
        new_cost = cost
        while time.time() - start_time < max_time - 1:
            last_cost = cost
            copy_routes = copy.deepcopy(routes)
            copy_remain = copy.deepcopy(remain)
            choice = random.random()
            # before = time.time()
            if choice < weight[0]:
                new_cost = cost + self_single_insertion(copy_routes, copy_remain)
            elif choice < weight[1]:
                new_cost = cost + cross_single_insertion(copy_routes, copy_remain)
            elif choice < weight[2]:
                new_cost = cost + swap(copy_routes, copy_remain)
            elif choice < weight[3]:
                new_cost = cost + cross_double_insertion(copy_routes, copy_remain)
            else:
                new_cost = cost + recombine(copy_routes, copy_remain)
            new_cost = flip(copy_routes, new_cost, copy_remain)
            # ha = get_hash(copy_routes)
            # if ha not in table:
            #     table.add(ha)
            #     break
            break
        if acceptance_probability(cost, new_cost, T, see) > random.random():
            # ha = get_hash(copy_routes)
            # if ha not in table:
            #     table.add(ha)
            #     cost = two_opt(copy_routes, copy_remain)
            # else:
            cost = new_cost
            # print(cost)
            # print(acceptance_probability(best_cost, new_cost, T))

            routes = copy_routes
            remain = copy_remain
            if cost < best_cost:
                best_cost = cost
                best_remain = remain
                best_routes = routes

        if cost == last_cost:
            repeat += 1
        if cost > 1.2 * best_cost:
            T = 10000
            cost = best_cost
            routes = best_routes
            remain = best_remain

        if repeat > 100:
            T = 10000
            repeat = 0
        T *= 1 - cool_rate
        # print(time.time()-before)

    # print(times)
    return best_routes, best_cost, best_remain


def self_single_insertion(routes, remain):
    r = random.randint(0, len(routes) - 1)
    route = routes[r]
    n = len(route)
    tid = random.randint(0, n - 1)
    if tid == 0:
        old_start = depot
    else:
        old_start = route[tid - 1][1]

    if tid == n - 1:
        old_end = depot
    else:
        old_end = route[tid + 1][0]
    task = route[tid]

    route.remove(task)
    idx = random.randint(0, n - 1)
    route.insert(idx, task)
    if idx == 0:
        new_start = depot
    else:
        new_start = route[idx - 1][1]
    if idx == n - 1:
        new_end = depot
    else:
        new_end = route[idx + 1][0]
    change = distance[old_start][old_end] - distance[new_start][new_end] + distance[new_start][task[0]] + \
             distance[task[1]][new_end] - distance[old_start][task[0]] - distance[task[1]][old_end]
    # print(f"new:{distance[old_start][task[0]]-distance[task[1]][old_end]}")
    # print(f"true:{cal_new_cost(route)}")
    # print(change+remain[r][1])
    # change=0
    # old_cost = remain[r][1]
    remain[r][1] += change
    return change


def cross_single_insertion(routes, remain):
    r = random.randint(0, len(routes) - 1)
    route = routes[r]
    tid = random.randint(0, len(route) - 1)
    task = route[tid]
    demand = task[3]
    candidates = []
    for i in range(len(routes)):
        if i != r and remain[i][0] >= demand:
            candidates.append(i)
    if len(candidates) == 0:
        return 0
    r2 = random.randint(0, len(candidates) - 1)
    route2 = routes[candidates[r2]]

    if tid == 0:
        old_start = depot
    else:
        old_start = route[tid - 1][1]

    if tid == len(route) - 1:
        old_end = depot
    else:
        old_end = route[tid + 1][0]

    idx = random.randint(0, len(route2))
    route2.insert(idx, task)
    route.remove(task)
    if idx == 0:
        new_start = depot
    else:
        new_start = route2[idx - 1][1]
    if idx == len(route2) - 1:
        new_end = depot
    else:
        new_end = route2[idx + 1][0]

    change1 = distance[old_start][old_end] - distance[old_start][task[0]] - distance[task[1]][old_end] - task[2]
    change2 = -distance[new_start][new_end] + distance[new_start][task[0]] + distance[task[1]][new_end] + task[2]
    if len(route) == 0:
        routes.remove(route)
    remain[r][0] += demand
    remain[candidates[r2]][0] -= demand
    remain[r][1] += change1
    remain[candidates[r2]][1] += change2
    # if change2 + change1 < 0:
    #     print(change1 + change2)
    if len(route) == 0:
        remain.pop(r)
    return change1 + change2


def cross_double_insertion(routes, remain):
    r = random.randint(0, len(routes) - 1)
    route = routes[r]
    if len(route) < 2:
        return 0
    tid = random.randint(0, len(route) - 2)
    task = route[tid]
    task2 = route[tid + 1]
    demand = task[3] + task2[3]
    candidates = []
    for i in range(len(routes)):
        if i != r and remain[i][0] >= demand:
            candidates.append(i)
    if len(candidates) == 0:
        return 0
    r2 = random.randint(0, len(candidates) - 1)
    route2 = routes[candidates[r2]]

    if tid == 0:
        old_start = depot
    else:
        old_start = route[tid - 1][1]

    if tid == len(route) - 2:
        old_end = depot
    else:
        old_end = route[tid + 2][0]

    idx = random.randint(0, len(route2))

    route2.insert(idx, task)
    route2.insert(idx + 1, task2)
    route.remove(task)
    route.remove(task2)

    if idx == 0:
        new_start = depot
    else:
        new_start = route2[idx - 1][1]
    if idx == len(route2) - 2:
        new_end = depot
    else:
        new_end = route2[idx + 2][0]

    #
    # old_cost1 = remain[r][1]
    # old_cost2 = remain[candidates[r2]][1]
    if len(route) == 0:
        # new_cost1 = cal_new_cost(route)
        # else:
        routes.remove(route)
        # new_cost1 = 0
    change1 = distance[old_start][old_end] - distance[old_start][task[0]] - distance[task2[1]][old_end] - task[2] - \
              task2[2] - distance[task[1]][task2[0]]
    change2 = -distance[new_start][new_end] + distance[new_start][task[0]] + distance[task2[1]][new_end] + task[2] + \
              task2[2] + distance[task[1]][task2[0]]
    remain[r][0] += demand
    remain[candidates[r2]][0] -= demand
    remain[r][1] += change1
    remain[candidates[r2]][1] += change2

    # print(f"true{cal_new_cost(route)}, {remain[r][1]}")
    if len(route) == 0:
        remain.pop(r)
    return change1 + change2


def acceptance_probability(old, new, T, see):
    if new < old:
        return 1.0
    else:
        # print((old - new) / T)
        return math.exp((old - new) * see / T)


def cal_new_cost(route):
    new_cost = 0
    new_cost += distance[depot][route[0][0]]
    for i in range(0, len(route)):
        if i == 0:
            start = depot
        else:
            start = route[i - 1][1]
        if i == len(route) - 1:
            end = depot
        else:
            end = route[i + 1][0]
        new_cost += distance[route[i][1]][end] + route[i][2]
    return new_cost


def get_ans(routes):
    ans = []
    for route in routes:
        ans.append(0)
        for task in route:
            ans.append((task[0], task[1]))
        ans.append(0)
    return ans


def two_opt(routes, remain):
    total = 0
    for i in range(len(routes)):
        route = routes[i]
        min = remain[i][1]
        best_route = route
        n = len(route)
        for a in range(0, n - 1):
            for b in range(a + 2, n):
                copy_route = copy.deepcopy(route)
                cost = reverse(copy_route, a, b)
                if cost < min:
                    # print(det)
                    best_route = copy_route
                    min = cost
                    remain[i][1] = cost
        routes[i] = best_route
        total += min
    # print(total)
    return total


def reverse(route, a, b):
    # r = 0
    # count = 0
    # global rv_total
    # global rv_ok
    # rv_total += 1
    # while True:
    #     r = random.randint(0, len(routes) - 1)
    #     route = routes[r]
    #     if len(route) >= 2 or count > 5:
    #         break
    #     else:
    #         count += 1
    # if count > 5:
    #     return 0
    # a, b = 0, 0
    # count = 0
    # while True:
    #     a = random.randint(0, len(route) - 1)
    #     b = random.randint(0, len(route) - 1)
    #     if abs(a - b) > 1 and a>0 and b>0:
    #         break
    #     else:
    #         count += 1
    # if count > 5:
    #     return 0

    # print("before")
    # print(len(route))
    # if a < b:
    # print(route)
    # print((a,b))
    # print(route)
    # print(route[a:b])
    # print(route[b - 1:a - 1:-1])
    # else:
    #     route[b:a] = route[a - 1:b - 1:-1]
    # print("after")
    # print(len(route))
    # print(route)
    # n_cost = cal_new_cost(route)
    # old_cost = remain[r][1]
    # remain[r][1] = n_cost
    # if n_cost - old_cost < 0:
    #     # print(f"reverse: {n_cost - old_cost}")
    #     rv_ok += 1
    # if a == 0:
    #     start = depot
    # else:
    #     start = route[a - 1][1]
    # if b == len(route):
    #     end = depot
    # else:
    #     end = route[b][0]
    # change = distance[route[a][1]][end] + distance[start][route[b - 1][0]] - distance[start][route[a][0]] - \
    #          distance[route[b - 1][1]][end]
    # old = cal_new_cost(route)
    # # print(a,b)
    # # print(route)
    route[a:b] = reversed(route[a:b])
    # print(route)
    new = cal_new_cost(route)
    # print(f"old{old}")
    # print(f"new{new - change}")
    return new


def swap(routes, remain):
    r1, r2 = 0, 0
    while True:
        r1 = random.randint(0, len(routes) - 1)
        r2 = random.randint(0, len(routes) - 1)
        if r1 != r2:
            break
    route1 = routes[r1]
    route2 = routes[r2]
    count = 0
    while True:
        r3 = random.randint(0, len(route1) - 1)
        r4 = random.randint(0, len(route2) - 1)
        temp1 = route2[r4]
        temp2 = route1[r3]

        if r3 == 0:
            old_start = depot
        else:
            old_start = route1[r3 - 1][1]
        if r3 == len(route1) - 1:
            old_end = depot
        else:
            old_end = route1[r3 + 1][0]

        if r4 == 0:
            new_start = depot
        else:
            new_start = route2[r4 - 1][1]
        if r4 == len(route2) - 1:
            new_end = depot
        else:
            new_end = route2[r4 + 1][0]

        new_demand1 = capacity - remain[r1][0] - temp2[3] + temp1[3]
        new_demand2 = capacity - remain[r2][0] - temp1[3] + temp2[3]
        # print(count)
        count += 1
        if (new_demand1 <= capacity and new_demand2 <= capacity) or count > len(route1) + len(route2):
            break
    new_route1 = route1[:r3] + [temp1] + route1[r3 + 1:]
    new_route2 = route2[:r4] + [temp2] + route2[r4 + 1:]
    if count > len(new_route1) + len(new_route2):
        return 0
    change1 = distance[old_start][temp1[0]] + distance[temp1[1]][old_end] + temp1[2] - distance[old_start][temp2[0]] - \
              distance[temp2[1]][old_end] - temp2[2]
    change2 = distance[new_start][temp2[0]] + distance[temp2[1]][new_end] + temp2[2] - distance[new_start][temp1[0]] - \
              distance[temp1[1]][new_end] - temp1[2]
    # new_cost1 = cal_new_cost(new_route1)
    # new_cost2 = cal_new_cost(new_route2)
    #
    # if new_cost2 != remain[r2][1] + change2:
    #     print("cnm")
    #
    #     print(f"true:{new_cost2}")
    #     print(remain[r2][1] + change2)
    # old_cost1 = remain[r1][1]
    # old_cost2 = remain[r2][1]
    routes[r1] = new_route1
    routes[r2] = new_route2
    remain[r1][1] += change1
    remain[r2][1] += change2
    remain[r1][0] = capacity - new_demand1
    remain[r2][0] = capacity - new_demand2
    # print(new_cost1 + new_cost2 - old_cost1 - old_cost2)
    return change2 + change1


def solver(s, weight):
    file_name, termination, see = arg_analysis()
    # print(termination)
    # random.seed(seed)
    global seed
    seed = see
    global max_time
    max_time = termination
    de, cap, dem, dis = read_file(file_name)
    global depot
    depot = de
    global capacity
    capacity = cap
    global distance
    distance = dis
    global table
    table = set()
    demands = dem
    random.seed(s)
    total, routes, remain = get_init(demands, weight)
    total = flip(routes, total, remain)
    two_opt(routes, remain)
    if weight[0] > 0.05:
        see = random.randint(10000, 12000)
    else:
        see = random.randint(8000, 10000)
    best_r, best_c, best_re = simulated_annealing(routes, total, remain, weight, see)
    t_cost = 0
    for route in best_r:
        t_cost += cal_new_cost(route)

    # print(f"weitht: {weight} , cost:{t_cost}, see{see}")
    ans = get_ans(best_r)
    return ans, t_cost


class Pro(mp.Process):
    def __init__(self, q1, q2):
        mp.Process.__init__(self, target=self.start)
        self.q1 = q1
        self.q2 = q2
        self.exit = mp.Event()

    def run(self):
        while True:
            s, weight = self.q1.get()
            r, c = solver(s, weight)
            self.q2.put((r, c))
            # print(time.time() - start_time)


if __name__ == "__main__":
    num = 8
    # print(mp.cpu_count())

    see = [random.randint(0, 10), seed, random.randint(11, 99), random.randint(100, 999), random.randint(1000, 9999),
           random.randint(10000, 99999),
           random.randint(100000, 500000), random.randint(500000, 9999999), random.randint(0, 10), seed,
           random.randint(11, 99), random.randint(100, 999),
           random.randint(1000, 9999), random.randint(10000, 99999),
           random.randint(100000, 500000), random.randint(500000, 9999999), random.randint(0, 10), seed,
           random.randint(11, 99), random.randint(100, 999), random.randint(1000, 9999),
           random.randint(10000, 99999),
           random.randint(100000, 500000), random.randint(500000, 9999999), random.randint(0, 10), seed,
           random.randint(11, 99), random.randint(100, 999),
           random.randint(1000, 9999), random.randint(10000, 99999),
           random.randint(100000, 500000), random.randint(500000, 9999999)]
    # single cross swap double recombine
    weight = [[0.2, 0.5, 0.7, 0.85], [0.05, 0.4, 0.6, 0.7], [0.05, 0.4, 0.6, 0.7], [0.2, 0.5, 0.7, 0.85],
              [0.2, 0.5, 0.7, 0.85], [0.05, 0.4, 0.6, 0.7], [0.05, 0.4, 0.6, 0.7], [0.2, 0.5, 0.7, 0.85],
              [0, 0.35, 0.9, 0.95], [0.5, 0.8, 0.95, 0.95], [0, 0, 0, 1],
              [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0.3, 0.8, 0.8, 0.9], [0.15, 0.35, 0.9, 0.95],
              [0.3, 0.6, 0.8, 0.9], [0.5, 0.8, 0.9, 0.95], [0.2, 0.7, 0.85, 0.95],
              [0.1, 0.3, 0.8, 0.9], [0.1, 0.25, 0.4, 0.9], [0.15, 0.3, 0.4, 0.5], [0.2, 0.4, 0.6, 0.8],
              [0, 0.35, 0.9, 0.95], [0.5, 0.8, 0.95, 0.95], [0, 0, 0, 1],
              [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0.3, 0.8, 0.8, 0.9]]

    # best_r, best_c = simulated_annealing(routes, total, remain)
    pro = []

    for i in range(num):
        pro.append(Pro(mp.Queue(), mp.Queue()))
        pro[i].start()
        pro[i].q1.put((see[i], weight[i]))
    result = []
    for i in range(num):
        result.append(pro[i].q2.get())
    list = sorted(result, key=lambda p: p[1])
    ans = list[0][0]
    cost = list[0][1]
    # print(time.time()-start_time)
    print("s " + str(ans).replace("[", "").replace("]", "").replace(" ", ""))
    print("q " + str(cost))
    # print(time.time() - start_time)
    for p in pro:
        p.terminate()
