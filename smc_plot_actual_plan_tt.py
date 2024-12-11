"""
This file contains the code for draw the plan and actual running timetable and calculate the delay time of each train.
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
                    plan_timetable.iloc[i, j] = minutes_to_hms(plan_timetable.iloc[i, j])

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

    # 读取数据
    path = './simulation_fast_inst/'
    # 如果不存在simulation_results文件夹，请先创建
    if not os.path.exists(path + '/plan_vs_actual'):
        os.mkdir(path + '/plan_vs_actual')

    DF = []

    # plan
    df_p = pd.read_excel(path + 'plan_tt.xlsx', index_col=0)
    df_p += 420
    DF.append(pre_deal_data(df_p))

    # fcfs的方案
    df_f = pd.read_excel(path + 'tt_fcfs.xlsx', index_col=0)
    df_f += 420
    DF.append(pre_deal_data(df_f))

    # pe-sega的方案
    df_model = pd.read_excel(path + 'tt_pesa_0.28.xlsx', index_col=0)
    df_model += 420
    DF.append(pre_deal_data(df_model))

    # model_-1的方案
    # df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_-1_tanh_d60_dense.xlsx', index_col=0)
    # df_model += 420
    # DF.append(pre_deal_data(df_model))

    # model_15的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_15_tanh_d60_dense.xlsx', index_col=0)
    df_model += 420
    DF.append(pre_deal_data(df_model))

    # model_10的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_10_tanh_d60_dense.xlsx', index_col=0)
    df_model += 420
    DF.append(pre_deal_data(df_model))

    # model_5的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_5_tanh_d60_dense.xlsx', index_col=0)
    df_model += 420
    DF.append(pre_deal_data(df_model))

    # model_0的方案
    df_model = pd.read_excel(path + 'tt_rl_10_9_3_60Rou_0_tanh_d60_dense.xlsx', index_col=0)
    df_model += 420
    DF.append(pre_deal_data(df_model))

    names = ['FCFS', 'PE-SEGA', 'policy_15', 'policy_10', 'policy_5', 'policy_0']
    names_label = ['FCFS', 'PE-SEGA', 'policy $\\rho=15\%$', 'policy $\\rho=10\%$', 'policy $\\rho=5\%$', 'policy $\\rho=0\%$']

    min_dur = [7, 8, 7, 5, 7, 8, 8, 7, 7]
    min_wait = pd.read_csv(path + 'min_wait_dense.csv', index_col=0)
    min_wait.columns = df_f.columns

    delay_o = df_model.iloc[0] - df_p.iloc[0]

    station_seq = ['JNX', 'TA', 'QFD', 'TZD', 'ZZ', 'XZD', 'SZD', 'BBN', 'DY', 'CZ']

    save = True
    first_station_time = []
    for each in df_model.iloc[0]:
        depart_t = minutes_to_hms(each)
        # 字符串变成datatime
        depart_t = datetime.datetime.strptime(depart_t, '%H:%M:%S')
        first_station_time.append(depart_t)
    first_station_time = pd.Series(first_station_time, index=delay_o.index)

    first_station_time_p = []
    for each in df_p.iloc[0]:
        depart_t = minutes_to_hms(each)
        # 字符串变成datatime
        depart_t = datetime.datetime.strptime(depart_t, '%H:%M:%S')
        first_station_time_p.append(depart_t)
    first_station_time_p = pd.Series(first_station_time_p, index=delay_o.index)

    DELAY = []
    for ith, each in enumerate(DF):
        if ith == 0:
            continue
        df, schedule = each
        simulator = TrainSimulator(schedule, low=0, high=0, min_dur=min_dur, min_wait=min_wait, ini_delay=None,
                                   station_seq=station_seq)
        simulator.simulate()
        simulator.plot_schedule(draw_planned=True,
                                draw_actual=False,
                                dynamic_y=True,
                                save_fig=save,
                                file_name=None,
                                line_type='-',
                                first_station_time=first_station_time)  # draw_planned=True 绘制计划图
        stations_offsets_p = simulator.stations_offsets  # 计划图的起点和终点

        # 根据当前的fig，继续画出实际运行图

        df_p, schedule_p = DF[0]
        delays = []
        simulator = TrainSimulator(schedule_p, low=0, high=0, fig_path=path, min_dur=min_dur, min_wait=min_wait, ini_delay=None,
                                   station_seq=station_seq)
        simulator.simulate()
        simulator.continue_plot = True
        simulator.stations_offsets = stations_offsets_p  # 实际运行图的起点和终点与计划图一致
        actual_times_df = simulator.plot_schedule(draw_planned=True,
                                                  draw_actual=False,
                                                  dynamic_y=True,
                                                  save_fig=save,
                                                  file_name=names_label[ith-1]+'.png',
                                                  file_name2=names[ith-1]+'.png',
                                                  line_type=':',
                                                  first_station_time=first_station_time_p)  # draw_planned=True 绘制计划图
        simulator.continue_plot = False

        # 统计列车延误时间
        total_delay_time = get_total_delay_time(actual_times_df, df.iloc[1:])
        delays.append(total_delay_time)


