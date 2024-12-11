"""
This file contains the code for analyzing the simulation results.
"""


import datetime
import numpy as np
import pandas as pd

def get_total_delay_time(real_timetable2, plan_timetable2, types='value'):

    # 按照到计划列车的顺序安排实际列车运行方案
    real_timetable2 = real_timetable2[plan_timetable2.columns]
    # 计算总的到站延误时间
    real_timetable = real_timetable2.fillna(datetime.time(0, 0))
    plan_timetable = plan_timetable2.fillna(datetime.time(0, 0))
    reference_date = datetime.date(2000, 1, 1)  # 参考日期
    if types == 'array':
        total_delay_time = np.zeros(real_timetable.shape)
    else:
        total_delay_time = []
    for i in range(real_timetable.shape[0]):
        if i % 2 != 0:    ## 只是算了列车的到站延误时间，所以只需要处理奇数行
            for j in range(real_timetable.shape[1]):
                # 将 time 对象与参考日期结合成 datetime 对象
                if type(plan_timetable.iloc[i, j]) is np.int64 or type(plan_timetable.iloc[i, j]) is int:
                    plan_timetable.iloc[i, j] = minutes_to_hms(float(plan_timetable.iloc[i, j]))

                if type(real_timetable.iloc[i, j]) is str:
                    real_timetable.iloc[i, j] = datetime.datetime.strptime(real_timetable.iloc[i, j], '%H:%M:%S').time()
                if type(plan_timetable.iloc[i, j]) is str:
                    plan_timetable.iloc[i, j] = datetime.datetime.strptime(plan_timetable.iloc[i, j], '%H:%M:%S').time()
                dt1 = datetime.datetime.combine(reference_date, real_timetable.iloc[i, j])
                dt2 = datetime.datetime.combine(reference_date, plan_timetable.iloc[i, j])

                delay = (dt1.timestamp() - dt2.timestamp())/60  # 计算两时间差的分钟数

                if types == 'array':
                    total_delay_time[i, j] = delay  # 计算两时间差的分钟数
                else:
                    total_delay_time.append(delay)  # 计算两时间差的分钟数
    if types == 'value':
        return np.sum(total_delay_time)
    else:
        return total_delay_time


def pre_deal_data(df):
    # 处理数据
    df.index = [f'{i // 2 + 1}' for i in range(df.shape[0])]
    df.columns = [f'T{i+1}' for i in range(df.shape[1])]
    schedule = convert_df_to_schedule(df, now=0)
    return df, schedule


if __name__ == '__main__':
    from simulation_fast import convert_df_to_schedule, minutes_to_hms, find_departure_times, TrainSimulator
    import os
    import random
    import copy


    # 固定随机种子
    seed = 42
    np.random.seed(seed)  # NumPy
    random.seed(seed)


    # 读取数据
    path = './simulation_fast_inst/'
    # 如果不存在simulation_results文件夹，请先创建
    if not os.path.exists(path + '/simulation_results'):
        os.mkdir(path + '/simulation_results')

    DF = []

    # fcfs的方案
    df_f = pd.read_excel(path + 'tt_fcfs.xlsx', index_col=0)
    DF.append(pre_deal_data(df_f))


    # pe-sega的方案
    df_model = pd.read_excel(path + 'tt_pesa_0.28.xlsx', index_col=0)
    DF.append(pre_deal_data(df_model))

    # # model_-1的方案
    # df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_-1_tanh_d60_dense.xlsx', index_col=0)
    # DF.append(pre_deal_data(df_model))

    # model_15的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_15_tanh_d60_dense.xlsx', index_col=0)
    DF.append(pre_deal_data(df_model))

    # model_10的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_10_tanh_d60_dense.xlsx', index_col=0)
    DF.append(pre_deal_data(df_model))

    # model_5的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_5_tanh_d60_dense.xlsx', index_col=0)
    DF.append(pre_deal_data(df_model))

    # model_0的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_0_tanh_d60_dense.xlsx', index_col=0)
    DF.append(pre_deal_data(df_model))

    names = ['FCFS', 'PE-SEGA', 'policy_15', 'policy_10', 'policy_5', 'policy_0']
    names_label = ['FCFS\n($\\rho_a=7\%$)', 'PE-SEGA\n($\\rho_a=28\%$)', 'policy-15%\n($\\rho_a=16.5\%$)',
                   'policy-10%\n($\\rho_a=11.2\%$)', 'policy-5%\n($\\rho_a=2.2\%$)',
                   'policy-0%\n($\\rho_a=-1\%$)']
    # names = ['FCFS']

    min_dur = [7, 8, 7, 5, 7, 8, 8, 7, 7]
    min_wait = pd.read_csv(path + 'min_wait_dense.csv', index_col=0)
    min_wait.columns = df_f.columns

    save = False
    draw_planned = False
    draw_actual = False
    show_box = True

    DELAY = []

    for each_simu in range(50):
        delays = []
        for ith, each in enumerate(DF):
            df, schedule = each
            if ith == 0:
                ini_delay = None
            else:
                ini_delay = copy.deepcopy(simulator.ini_delay)

            simulator = TrainSimulator(schedule, low=0, high=10, min_dur=min_dur, min_wait=min_wait,
                                       ini_delay=ini_delay)
            simulator.simulate()
            actual_times_df = simulator.plot_schedule(draw_planned=draw_planned,
                                                      draw_actual=draw_actual,
                                                      dynamic_y=True,
                                                      save_fig=save,
                                                      file_name=names[ith],
                                                      line_type='--')  # draw_planned=True 绘制计划图

            # 请输出运行结果
            # print(actual_times_df)  # 输出实际运行时间

            # 保存实际运行时间到csv文件
            if save:
                actual_times_df.to_csv(path + '/simulation_results/' + 'actual_schedule_'+names[ith]+str(each_simu+1)+'.csv')
            print(f"模拟{each_simu+1}次完成")

            # 统计列车延误时间
            total_delay_time = get_total_delay_time(actual_times_df, df.iloc[1:])
            delays.append(total_delay_time)
        DELAY.append(delays)

    if show_box:
        # 画出盒子图
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # 设置全局字体为 Times New Roman，大小为 14
        # rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
        rcParams['font.size'] = 18  # 设置字体大小为 14
        # box  = plt.boxplot(np.array(DELAY))
        plt.figure(figsize=(10, 6))
        box = plt.boxplot(np.array(DELAY), patch_artist=True, widths=0.8)
        # 设置颜色 (浅色)
        colors = ['#E6F7FF', '#FFEBCC', '#FFCCCC', '#DFF0D8', '#F9E79F', '#D5DBDB']

        # 遍历箱体并设置颜色
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)  # 设置箱体颜色
            patch.set_edgecolor('black')  # 设置边框颜色
            patch.set_linewidth(1.5)  # 设置边框线宽

        plt.ylim(0, 180)
        plt.xticks(range(1, len(names)+1), names_label)
        # plt.xlabel('policies with different $\\rho$')
        plt.ylabel('Total Delay Time (min)')
        # 保存柱状图
        # plt.savefig('boxplot.png')
        plt.show(block=True)


