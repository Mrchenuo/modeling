import xlrd  # 导入库
import numpy as np
import pandas as pd
import random
import xlsxwriter
from load_model import ans3_pred
from load_model import ans2_pred

population_size = 1974  # 种群大小
# population_size = 38  # 种群大小
chromosome_size = 729  # 染色体长度
generation_size = 6  # 最大迭代次数
cross_rate = 0.7  # 交叉概率
mutate_rate = 0.1  # 变异概率
best_number = 100  # 保留的最佳个体的数量(best_number需要大于1)

# 打开文件
# xls = pd.read_excel(r"D:\GitHub\modeling\D\ans2\Molecular_Descriptor_t.xlsx", engine='openpyxl', sheet_name='training')
xls = pd.read_excel(r"D:\GitHub\modeling\D\ans3\Molecular_Descriptor.xlsx", engine='openpyxl', sheet_name='training')
xls = xls.iloc[:, 1:].values

# 向量矩阵初始化
best_fitness = np.zeros(best_number)
best_generation = np.zeros(best_number)
fitness_value = np.zeros(population_size)
fitness_sum = np.zeros(population_size)
temp_chromosome = np.zeros(chromosome_size)
fitness_average = np.zeros(generation_size)
population_new = np.zeros((population_size, chromosome_size))
population = np.zeros((population_size, chromosome_size))
best_individual = np.zeros((best_number, chromosome_size))

# 初始种群生成
for i in range(population_size):
    for j in range(chromosome_size):
        population[i, j] = xls[i, j]

# 迭代
for G in range(generation_size):
    # TODO:返回二三问预测值
    pred_list_2 = ans2_pred(population)
    pred_list_3 = ans3_pred(population)
    # 适应度计算
    for i in range(population_size):
        fitness_value[i] = random.uniform(0, 1)

    # 选择
    for i in range(population_size):
        min_index = i
        for j in range(i + 1, population_size):
            if fitness_value[j] < fitness_value[min_index]:
                min_index = j
        if min_index != i:
            temp = fitness_value[i]
            fitness_value[i] = fitness_value[min_index]
            fitness_value[min_index] = temp
            for k in range(chromosome_size):
                temp_chromosome[k] = population[i, k]
                population[i, k] = population[min_index, k]
                population[min_index, k] = temp_chromosome[k]
    for i in range(population_size):
        if i == 1:
            fitness_sum[i] = fitness_value[i]
        else:
            fitness_sum[i] = fitness_sum[i - 1] + fitness_value[i]
    fitness_average[G] = fitness_sum[population_size - 1] / population_size

    # 筛选出最佳的前best_number个个体
    if G != 0:
        for i in range(1, best_number + 1):
            if fitness_value[population_size - i] > best_fitness[0]:
                for j in range(1, best_number):
                    if fitness_value[population_size - i] > best_fitness[j]:
                        if j == best_number - 1:
                            for k in range(best_number - 1):
                                best_generation[k] = best_generation[k + 1]
                                best_fitness[k] = best_fitness[k + 1]
                                for s in range(chromosome_size):
                                    best_individual[k, s] = best_individual[k + 1, s]
                            best_generation[best_number - 1] = G + 1
                            best_fitness[best_number - 1] = fitness_value[population_size - i]
                            for k in range(chromosome_size):
                                best_individual[best_number - 1, k] = population[population_size - i, k]
                        else:
                            continue
                    else:
                        for k in range(j - 2):
                            best_generation[k] = best_generation[k + 1]
                            best_fitness[k] = best_fitness[k + 1]
                            for s in range(chromosome_size):
                                best_individual[k, s] = best_individual[k + 1, s]
                        best_generation[j - 1] = G + 1
                        best_fitness[j - 1] = fitness_value[population_size - i]
                        for k in range(chromosome_size):
                            best_individual[j - 1, k] = population[population_size - i, k]
                        break
            else:
                break
    else:
        for i in range(best_number):
            best_fitness[i] = fitness_value[population_size - best_number + i]
            for j in range(chromosome_size):
                best_individual[i, j] = population[population_size - best_number + i, j]

    # 染色体交叉
    for i in range(population_size):
        r = random.uniform(0, 1) * fitness_sum[population_size - 1]
        first = 0
        last = population_size - 1
        mid = round((last + first) / 2)
        idx = -1
        while first <= last & idx == -1:
            if r > fitness_sum[mid]:
                first = mid
            elif r < fitness_sum[mid]:
                last = mid
            else:
                idx = mid
                break
            mid = round((last + first) / 2)
            if (last - first) == 1:
                idx = last
                break
        for j in range(chromosome_size):
            population_new[i, j] = population[idx, j]
    for i in range(population_size):
        for j in range(chromosome_size):
            population[i, j] = population_new[i, j]
    for i in range(0, population_size, 2):
        if random.uniform(0, 1) < cross_rate:
            cross_position = round(random.uniform(0, chromosome_size - 1))
            for j in range(cross_position, chromosome_size):
                temp = population[i, j]
                population[i, j] = population[i + 1, j]
                population[i + 1, j] = temp

    # 变异
    for i in range(population_size):
        if random.uniform(0, 1) < mutate_rate:
            mutate_position = round(random.uniform(0, chromosome_size - 1))
            mutate_value = round(random.uniform(0, population_size - 1))
            population[i, mutate_position] = xls[mutate_value, mutate_position]

print("best_fitness:", best_fitness)
# print("best_generation:", best_generation)

# 写进excel
workbook = xlsxwriter.Workbook('Excel_test.xlsx')
worksheet = workbook.add_worksheet()
for i in range(best_number):
    worksheet.write(i + 1, 1, best_fitness[i])
    # worksheet.write(i + 1, 2, best_generation[i])
    for j in range(chromosome_size):
        worksheet.write(i + 1, j + 3, best_individual[i, j])
workbook.close()
