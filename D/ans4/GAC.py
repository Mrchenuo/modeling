import datetime

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
generation_size = 200  # 最大迭代次数
cross_rate = 0.6  # 交叉概率
mutate_rate = 0.01  # 变异概率
best_number = 100  # 保留的最佳个体的数量(best_number需要大于1)
admet = 3  # 对ADMET性质的要求

# 打开文件
xls = pd.read_excel(r"D:\GitHub\modeling\D\ans2\Molecular_Descriptor.xlsx", engine='openpyxl', sheet_name='training')
# xls = pd.read_excel(r"Molecular_Descriptor.xlsx", engine='openpyxl', sheet_name='training')
xls = xls.iloc[:, 1:].values

# 向量矩阵初始化
best_fitness = np.zeros(best_number)
best_generation = np.zeros(best_number)
fitness_value = np.zeros(population_size)
fitness_sum = np.zeros(population_size)
temp_chromosome = np.zeros(chromosome_size)
fitness_average = np.zeros(generation_size)
# pred_list_3_sum = np.zeros(population_size)
temp_pred3 = np.zeros(5)
best_pred3 = np.zeros((5, best_number))
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

    # pred_list_2_3 = np.zeros(population_size)
    # pred_list_3_3 = np.ones((5, population_size))
    pred_list_2 = np.concatenate(pred_list_2)
    pred_list_3 = np.array(pred_list_3)

    # 适应度计算
    count = 0
    for i in range(population_size):
        for j in range(5):
            if pred_list_3[j, i] == 0:
                if j == 1 or j == 2 or j == 4:
                    count = count + 1
            elif j == 0 or j == 3:
                count = count + 1
        if count >= admet:
            fitness_value[i] = pred_list_2[i]
        else:
            fitness_value[i] = 0
        count = 0

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
            for k in range(5):
                temp_pred3[k] = pred_list_3[k, i]
                pred_list_3[k, i] = pred_list_3[k, min_index]
                pred_list_3[k, min_index] = temp_pred3[k]
            for k in range(chromosome_size):
                temp_chromosome[k] = population[i, k]
                population[i, k] = population[min_index, k]
                population[min_index, k] = temp_chromosome[k]
    for i in range(population_size):
        if i == 0:
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
                                for s in range(5):
                                    best_pred3[s, k] = best_pred3[s, k + 1]
                                for s in range(chromosome_size):
                                    best_individual[k, s] = best_individual[k + 1, s]
                            best_generation[best_number - 1] = G + 1
                            best_fitness[best_number - 1] = fitness_value[population_size - i]
                            for k in range(5):
                                best_pred3[k, best_number - 1] = pred_list_3[k, population_size - i]
                            for k in range(chromosome_size):
                                best_individual[best_number - 1, k] = population[population_size - i, k]
                        else:
                            continue
                    else:
                        for k in range(j - 2):
                            best_generation[k] = best_generation[k + 1]
                            best_fitness[k] = best_fitness[k + 1]
                            for s in range(5):
                                best_pred3[s, k] = best_pred3[s, k + 1]
                            for s in range(chromosome_size):
                                best_individual[k, s] = best_individual[k + 1, s]
                        best_generation[j - 1] = G + 1
                        best_fitness[j - 1] = fitness_value[population_size - i]
                        for k in range(5):
                            best_pred3[k, j - 1] = pred_list_3[k, population_size - i]
                        for k in range(chromosome_size):
                            best_individual[j - 1, k] = population[population_size - i, k]
                        break
            else:
                break
    else:
        for i in range(best_number):
            best_fitness[i] = fitness_value[population_size - best_number + i]
            for j in range(5):
                best_pred3[j, i] = pred_list_3[j, population_size - best_number + i]
            for j in range(chromosome_size):
                best_individual[i, j] = population[population_size - best_number + i, j]

    # # 退火
    # for i in range(population_size):
    #     for j in range(chromosome_size):
    #         if random.uniform(0, 1) < SA_rate:
    #             population[i, j] = xls[i, j]

    # 染色体交叉
    for i in range(population_size):
        r = random.uniform(0, 1) * fitness_sum[population_size - 1]
        first = 0
        last = population_size - 1
        mid = round((last + first) / 2)
        idx = -1
        while first <= last and idx == -1:
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
            cross_position1 = round(random.uniform(0, chromosome_size - 1))
            cross_position2 = round(random.uniform(0, chromosome_size - 1))
            for j in range(min(cross_position1, cross_position2), max(cross_position1, cross_position2)):
                temp = population[i, j]
                population[i, j] = population[i + 1, j]
                population[i + 1, j] = temp

    # 变异
    for i in range(population_size):
        for j in range(chromosome_size):
            if random.uniform(0, 1) < mutate_rate:
                mutate_value = round(random.uniform(0, population_size - 1))
                population[i, j] = xls[mutate_value, j]

print("best_fitness:", best_fitness)
print("best_generation:", best_generation)

# 写进excel
ISOTIMEFORMAT = '%Y%m%d%H%M%S'
time_str = datetime.datetime.now().strftime(ISOTIMEFORMAT)
excel_name = time_str + "_cr_" + str(cross_rate) + "_mr_" + str(mutate_rate) + "_" + str(admet) + '.xlsx'
workbook = xlsxwriter.Workbook(excel_name)
worksheet = workbook.add_worksheet()
for i in range(best_number):
    worksheet.write(i + 1, 1, best_fitness[i])
    worksheet.write(i + 1, 2, best_generation[i])
    for j in range(5):
        worksheet.write(i + 1, j + 3, best_pred3[j, i])
    for j in range(chromosome_size):
        worksheet.write(i + 1, j + 8, best_individual[i, j])
workbook.close()
