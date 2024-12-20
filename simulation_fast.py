import heapq
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
import copy
import pandas as pd
from itertools import count
from collections import defaultdict

# 全局计数器
event_counter = count()


class TrainSimulator:
    def __init__(self, schedule, low=0, high=5, min_wait=None, min_dur=None, station_seq=None, fig_path=None, ini_delay=None):
        """
        初始化列车仿真器
        :param schedule: 列车时刻表，格式为列表，每个元素是一个字典
                         包含 train_id、start、end、departure_time、arrival_time
        """
        self.schedule = schedule
        self.events = []  # 事件优先队列
        self.current_time = datetime.now()
        self.trains = {}  # 记录每辆列车的状态
        self.logs = []  # 用于绘图的日志记录
        self.low = low  # 最低延误分钟数
        self.high = high  # 最高延误分钟数
        self.continue_plot = False  # 连续绘图标志位
        self.fig_path = fig_path
        self.min_dur = min_dur
        self.min_wait = min_wait
        self.station_seq = station_seq
        if ini_delay is None:
            self.ini_delay = []
            self.delay_load = True
        else:
            self.ini_delay = pd.DataFrame(ini_delay)
            self.delay_load = False

        # 随机冲schedule里选一个
        self.event = np.random.choice(schedule)
        # 初始化事件队列
        for item in schedule:
            train_id = item["train_id"]
            dep_time = datetime.strptime(item["departure_time"], "%H:%M:%S")
            # 离开时间里加入随机的秒，避免多个列车同时发车
            # dep_time += timedelta(seconds=random.randint(1, 59))
            arr_time = datetime.strptime(item["arrival_time"], "%H:%M:%S")
            sequence = next(event_counter)  # 获取当前事件的序列号
            heapq.heappush(self.events, (dep_time, sequence, "depart", train_id, item))
            # 如果self.trains中不存在train_id，则添加一个新的列车状态
            if train_id not in self.trains:
                self.trains[train_id] = {
                    "current_section": None,
                    "next_departure_time": dep_time,
                    "next_arrival_time": arr_time,
                    "status": "waiting",
                }

        # while self.events:
        #     # 取出最早的事件
        #     event_time, event_type, train_id, event_data = heapq.heappop(self.events)
        #     print(f"当前时间：{event_time}, 事件：{event_type}, 列车：{train_id}, 事件数据：{event_data}")
        #
        # print('hello')

    def log_event_queue(self, event_queue, train_id=None):
        print("Current Event Queue:")
        for event in sorted(event_queue):
            if train_id is not None:
                if event[3] == train_id:
                    print(event)
            else:
                print(event)

    def simulate(self):
        """
        开始仿真
        """
        while self.events:
            # 取出最早的事件
            event_time, _,  event_type, train_id, event_data = heapq.heappop(self.events)
            self.current_time = event_time
            print(f"当前时间：{self.current_time}, 事件：{event_type}, 列车：{train_id}, 事件数据：{event_data}")
            self.log_event_queue(self.events, 'T6')   # debug use
            # 当运行到这个列车的事件后，所有后续列车全部都删除
            self.events = [ev for ev in self.events if not ev[3] == train_id]

            # 使用 heapify 重新构建堆
            heapq.heapify(self.events)

            if event_type == "depart":
                self.handle_departure(train_id, event_data)
            elif event_type == "arrive":
                self.handle_arrival(train_id, event_data)

    def handle_departure(self, train_id, event_data):
        """
        处理列车发车事件
        """
        section = (event_data["start"], event_data["end"])
        planned_departure_time = datetime.strptime(event_data["departure_time"], "%H:%M:%S")
        planned_arrival_time = datetime.strptime(event_data["arrival_time"], "%H:%M:%S")
        planned_operation_time = planned_arrival_time - planned_departure_time

        # 检查发车间隔约束（与前一列车至少相隔 3 分钟）
        planned_headway = copy.deepcopy(planned_departure_time)
        for log in reversed(self.logs):
            if log["event"] == "depart" and log["station"] == section[0]:
                last_depart_time = log["time"]
                if planned_departure_time < last_depart_time + timedelta(minutes=3):
                    planned_departure_time = last_depart_time + timedelta(minutes=3)

                # 同时保证列车的到站时间满足越行约束
                for item in self.events:
                    if item[3] == log["train_id"] and item[2] == "arrive" and item[4]["end"] == section[1]:
                        planned_headway = item[0] + timedelta(minutes=3)
                break

        print(
            f"{planned_departure_time} - 列车 {train_id} 从 {section[0]} 发车，开往 {section[1]}"
        )

        # 记录发车时间
        entry = find_departure_times(self.schedule, train_id, start_station=section[0])  # 找到schedule中对应的train_id对应的start和end的departure_time
        self.logs.append({
            "train_id": train_id,
            "event": "depart",
            "time": planned_departure_time,
            "planned_time": entry["departure_time"],
            "station": section[0],
        })

        # 更新列车状态
        self.trains[train_id]["current_section"] = section
        self.trains[train_id]["status"] = "on the way"

        # 计算实际到达时间（加上一个随机干扰）
        # delay = random.randint(self.low, self.high)  # 随机延误，单位分钟
        # 加入服从对数正态分布的随机干扰
        # if self.event['train_id'] == train_id and self.event['start'] == section[0]:
        #     # delay = np.random.lognormal(2.76695, 1.18678)//60  # 单位分钟
        #     delay = random.randint(self.low, self.high)  # 随机延误，单位分钟
        #     if delay < self.low:
        #         delay = self.low
        #     elif delay > self.high:
        #         delay = self.high
        #     print('##### train {} will delay in section {} {} minutes'.format(self.event['train_id'],
        #                                                                       self.event['start'], delay))
        # else:
        #     delay = 0.

        if self.delay_load:
            delay = np.random.lognormal(2.76695, 1.18678) // 60  # 单位分钟
            self.ini_delay.append({
                "train_id": train_id,
                "event": "delay",
                "time": delay,
                "station": section[1],
            })
        else:
            # find the delay from the ini_delay
            filtered = self.ini_delay[(self.ini_delay['train_id'] == train_id) & (self.ini_delay['station'] == section[1])]
            delay = filtered['time'].iloc[0] if not filtered.empty else None

        minimum_operation_time = self.min_dur[int(section[0])-1]

        # 实际区间运行时间，
        # if delay == 0:
        actual_arrival_time = np.max([planned_headway, planned_departure_time + timedelta(minutes=minimum_operation_time), planned_arrival_time])
        actual_arrival_time += timedelta(minutes=delay)
        # else:
        #     actual_arrival_time = planned_departure_time + planned_operation_time + timedelta(minutes=delay)  # 按照计划来跑，不存在最短运行时间，赶点的情况
        #     # 保证列车下一站的到站时间大于计划时间，
        #     if actual_arrival_time < planned_headway:
        #         actual_arrival_time = planned_headway

        # 检查到站间隔约束（与前一列车至少相隔 3 分钟）
        for log in reversed(self.logs):
            if log["event"] == "arrive" and log["station"] == section[1]:
                last_arrival_time = log["time"]
                if actual_arrival_time < last_arrival_time + timedelta(minutes=3):
                    actual_arrival_time = last_arrival_time + timedelta(minutes=3)
                break

        # 添加到站事件
        # actual_arrival_time += timedelta(seconds=random.randint(1, 59))
        sequence = next(event_counter)  # 获取当前事件的序列号
        heapq.heappush(self.events, (actual_arrival_time, sequence, "arrive", train_id, event_data))
        self.trains[train_id]["next_arrival_time"] = actual_arrival_time

    def handle_arrival(self, train_id, event_data):
        """
        处理列车到站事件
        """
        section = (event_data["start"], event_data["end"])
        print(
            f"{self.current_time.strftime('%H:%M:%S')} - 列车 {train_id} 到达 {section[1]}"
        )

        entry = find_departure_times(self.schedule, train_id, end_station=section[1])

        # 记录到站时间
        self.logs.append({
            "train_id": train_id,
            "event": "arrive",
            "time": self.current_time,
            "planned_time": entry["arrival_time"],
            "station": section[1],
        })

        # 更新列车状态
        self.trains[train_id]["current_section"] = None
        self.trains[train_id]["status"] = "waiting"

        # 确保最短停站时间（2 分钟）
        wait = self.min_wait[train_id].iloc[int(section[1])-1]
        next_departure_time = self.current_time + timedelta(minutes=float(wait))

        # 根据当前列车编号以及当前车站，查找下一个区间的对应的发车事件
        for item in self.schedule:
            if item["train_id"] == train_id and item["start"] == section[1]:
                # 检查是否还有后续区间
                next_start = item["start"]
                next_end = item["end"]
                planned_next_departure = datetime.strptime(item["departure_time"], "%H:%M:%S")
                planed_next_arrival = datetime.strptime(item["arrival_time"], "%H:%M:%S")
                planed_next_operation_time = planed_next_arrival - planned_next_departure

                if next_departure_time > planned_next_departure:
                    planned_next_departure = next_departure_time

                # 检查是否与前车发车间隔符合要求（3分钟）
                for log in reversed(self.logs):
                    if log["event"] == "depart" and log["station"] == next_start:
                        last_depart_time = log["time"]
                        if planned_next_departure < last_depart_time + timedelta(minutes=3):
                            planned_next_departure = last_depart_time + timedelta(minutes=3)
                        break
                minimum_operation_time = self.min_dur[int(section[1])-1]
                planed_next_arrival = np.max([timedelta(minutes=float(minimum_operation_time)) + planned_next_departure, planed_next_arrival])
                # 保证新生成的事件的到站时间满足越行约束
                for events in reversed(self.events):
                    if events[2] == "depart" and events[4]["end"] == next_end and events[3] != train_id:
                        if events[0] < planned_next_departure:
                            arrival_time = datetime.strptime(events[4]["arrival_time"], "%H:%M:%S")
                            if planed_next_arrival < arrival_time + timedelta(minutes=3):
                                planed_next_arrival = arrival_time + timedelta(minutes=3)
                            break

                # 更新事件队列中相关的发车事件
                sequence = next(event_counter)  # 获取当前事件的序列号
                # planned_next_departure += timedelta(seconds=random.randint(1, 59))
                new_event = (planned_next_departure, sequence, "depart", train_id, {
                    "start": next_start,
                    "end": next_end,
                    "departure_time": planned_next_departure.strftime("%H:%M:%S"),
                    "arrival_time": planed_next_arrival.strftime("%H:%M:%S"),
                })

                # 过滤掉旧的发车事件并插入新的事件
                self.events = [
                    ev for ev in self.events if not (ev[2] == "depart" and ev[3] == train_id)
                ]
                # 使用 heapify 重新构建堆
                heapq.heapify(self.events)
                heapq.heappush(self.events, new_event)

                break

    def plot_schedule(self, draw_planned=True, draw_actual=True, dynamic_y=True, save_fig=False, file_name2=None,
                      file_name='train_schedule.png', line_type='-', first_station_time=None):
        """
        绘制列车运行时刻表，包括计划图和实际图，同时标记延误信息。
        同一列车的计划图和实际图使用一致的颜色。
        """
        if (draw_planned or draw_actual) and not self.continue_plot:
             plt.figure(figsize=(14, 8))
        stations = list(set([event["station"] for event in self.logs]))
        stations = np.sort([int(stations[i]) for i in range(len(stations))])
        stations = [str(i) for i in stations]
        train_ids = sorted(list(set([event["train_id"] for event in self.logs])))
        # 按照列车名字后面的数字进行排序
        train_ids = sorted(train_ids, key=lambda x: int(x.split('T')[-1]))

        if dynamic_y and not self.continue_plot:
            # 动态计算区间长度
            avg_section_times = self.calculate_avg_section_times()
            stations_offsets = {stations[0]: 0}  # 第一个站点的偏移量为 0
            for i in range(1, len(stations)):
                section = (stations[i - 1], stations[i])
                offset = avg_section_times.get(section, 10)  # 默认最小间隔 10
                stations_offsets[stations[i]] = stations_offsets[stations[i - 1]] + offset
            self.stations_offsets = stations_offsets

        if self.continue_plot and hasattr(self, "stations_offsets"):
            stations_offsets = self.stations_offsets

        # 为每辆列车分配固定颜色
        colors = list(mcolors.TABLEAU_COLORS.values())
        train_colors = {train_id: colors[i % len(colors)] for i, train_id in enumerate(train_ids)}

        actual_times_df = pd.DataFrame()
        for train_id in train_ids:
            # 实际运行数据
            actual_times = []
            actual_times_hms = []
            actual_positions = []
            delays = []  # 延误时间记录

            # 计划运行数据
            planned_times = []
            planned_positions = []

            for event in self.logs:
                if event["train_id"] == train_id:
                    actual_times.append(event["time"])
                    actual_times_hms.append(event["time"].strftime("%H:%M:%S"))

                    if not dynamic_y:
                        actual_positions.append(stations.index(event["station"]))
                    else:
                        actual_positions.append(stations_offsets[event["station"]])

                    # 计算延误时间（实际时间与计划时间的差值，单位分钟）
                    planned_time = datetime.strptime(event["planned_time"], "%H:%M:%S")
                    delay = (event["time"] - planned_time).total_seconds() / 60
                    delays.append(delay)
                    if len(planned_times) == 0 and first_station_time is not None:
                        planned_times.append(first_station_time[train_id])
                    planned_times.append(planned_time)
                    if not dynamic_y:
                        if len(planned_positions) == 0 and first_station_time is not None:
                            planned_positions.append(stations.index(event["station"]))
                        planned_positions.append(stations.index(event["station"]))
                    else:
                        if len(planned_positions) == 0 and first_station_time is not None:
                            planned_positions.append(stations_offsets[event["station"]])
                        planned_positions.append(stations_offsets[event["station"]])

            # 获取该列车的颜色
            train_color = train_colors[train_id]

            # 绘制计划图（虚线）
            if draw_planned:
                if train_id == 'T2' and not self.continue_plot :
                    linewidth = 3
                    plt.plot(planned_times, planned_positions, marker='o', markersize=10, linestyle=line_type, linewidth=linewidth,
                             color=train_color, label=f"{train_id}")
                else:
                    linewidth = 2
                    plt.plot(planned_times, planned_positions, linestyle=line_type, linewidth=linewidth,
                             color=train_color, label=f"{train_id}") # - plan

            # 绘制实际图（实线）
            if draw_actual:
                plt.plot(actual_times, actual_positions, marker='o', markersize=6, color=train_color, linewidth=2, label=f"{train_id} - actual")

            # 将实际到达时间转换为Series并添加到DataFrame中
            # 使用车站索引作为行索引，列车ID作为列名
            actual_times_series = pd.Series(actual_times_hms, index=actual_positions, name=train_id)
            # 如果actual_times_series的长度不够，则用nan填充
            if len(actual_times_series) < (len(stations)-1)*2:
                actual_times_series_2 = pd.Series(name=train_id, dtype=object).reindex_like(actual_times_df)
                # 把相同index的元素赋值给actual_times_series_2
                ss = 0
                tip = 1
                for ii in range(len(actual_times_series_2)):
                    if actual_times_series_2.index[ii] in actual_times_series.index:
                        if tip:
                            tip = 0
                        else:
                            actual_times_series_2.iloc[ii] = actual_times_series.iloc[ss]
                            ss += 1
                    else:
                        tip = 1

                actual_times_series = actual_times_series_2

            actual_times_df = pd.concat([actual_times_df, actual_times_series], axis=1)

            # 标记延误信息
            if draw_actual:
                for t, p, d in zip(actual_times, actual_positions, delays):
                    if d > 0:
                        plt.text(t, p, f"+{int(d)}'", color="red", fontsize=9, ha='left', va='bottom', fontweight='bold')
                    elif d < 0:
                        plt.text(t, p, f"{int(d)}'", color="blue", fontsize=9, ha='left', va='bottom', fontweight='bold')

        # 格式化横坐标时间
        if draw_planned or draw_actual and not self.continue_plot:
            plt.rcParams['font.size'] = 12
            min_time = min(event["time"] for event in self.logs).replace(minute=0, second=0, microsecond=0)
            max_time = max(event["time"] for event in self.logs).replace(minute=0, second=0, microsecond=0)
            # 取整点，
            time_range = pd.date_range(min_time, max_time + timedelta(minutes=50), freq="20min")  # 1分钟间隔
            plt.gca().set_xticks(time_range)
            plt.xticks(fontsize=20)  # 设置字体大小为 14
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # 10分钟间隔

            if not dynamic_y:
                plt.yticks(range(len(stations)), stations, font_size=12)
            else:
                plt.yticks([stations_offsets[station] for station in stations], stations)
            if file_name is not None:
                # 绘制图例
                plt.xlabel("Time", fontsize=20)
                plt.ylabel("Station", fontsize=20)
                plt.title(r"Rescheduled by {}".format(file_name[:-4]), fontsize=20)
                plt.legend(fontsize=12, loc='upper left', title="Train ID", title_fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.6)
                # plt.tight_layout()
                if self.station_seq is not None:
                    plt.gca().set_yticklabels(self.station_seq)
                    plt.gca().tick_params(axis='y', labelsize=14)

                if save_fig:
                    # 修改，将图片保存到plan_vs_actual里
                    plt.savefig(self.fig_path + 'plan_vs_actual/' + file_name2)
                    # plt.savefig(file_name)

                plt.show(block=True)
                plt.close()
        return actual_times_df

    def calculate_avg_section_times(self):
        """
        根据时刻表计算各区间的平均运行时间。
        """
        schedule = self.schedule

        section_times = defaultdict(list)

        for event in schedule:
            start = event["start"]
            end = event["end"]
            departure_time = datetime.strptime(event["departure_time"], "%H:%M:%S")
            arrival_time = datetime.strptime(event["arrival_time"], "%H:%M:%S")
            duration = (arrival_time - departure_time).total_seconds() / 60  # 转化为分钟
            section_times[(start, end)].append(duration)

        avg_section_times = {
            section: np.mean(times) for section, times in section_times.items()
        }
        return avg_section_times


def find_departure_times(schedule, train_id, start_station=None, end_station=None):
    for entry in schedule:
        if start_station:
            if entry["train_id"] == train_id and entry["start"] == start_station:
                return entry
        else:
            if entry["train_id"] == train_id and entry["end"] == end_station:
                return entry


def minutes_to_hms(minutes):
    # 将分钟转换为 timedelta 对象
    # 如果minites为float64类型，先转化为float类型，否则不做处理
    if pd.api.types.is_float_dtype(minutes):
        minutes = minutes.astype(float)

    # 如果数据为整形，则变为浮点型
    minutes = float(minutes)

    # 如果数据是nan，则返回nan
    if pd.isnull(minutes):
        return np.nan
    else:
        time_delta = timedelta(minutes=minutes)

        # 计算总秒数
        total_seconds = int(time_delta.total_seconds())

        # 计算小时、分钟和秒
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # 格式化为 "H:M:S" 字符串
        time_string = f"{hours:02}:{minutes:02}:{seconds:02}"

        return time_string


def convert_df_to_schedule(timetable_df, now=8*60):
    """
    将车站列和列车时刻表矩阵的 DataFrame 转换为指定的 schedule 格式。

    Args:
        timetable_df (pd.DataFrame): 输入的时刻表 DataFrame，行索引为车站，列索引为列车名。
                                      奇数行是到站时间，偶数行是发车时间。

    Returns:
        list: 转换后的 schedule 列表，每个元素是一个字典，表示一个列车区间。
    """
    schedule = []

    # 遍历每个列车
    for train_id in timetable_df.columns:
        # 获取列车的时刻
        train_schedule = timetable_df[train_id]

        # 遍历每个车站，生成相邻站之间的区间
        for i in range(1, len(train_schedule) - 1, 2):
            start_station = timetable_df.index[i]
            end_station = timetable_df.index[i + 1]
            departure_time = train_schedule.iloc[i]
            arrival_time = train_schedule.iloc[i + 1]

            # 如果deparure_time是数值，则不做处理，如果是字符串，则需要将字符串转化为分钟数
            if isinstance(departure_time, str):
                departure_time = datetime.strptime(departure_time, "%H:%M:%S").hour*60. + datetime.strptime(departure_time, "%H:%M:%S").minute
                arrival_time = datetime.strptime(arrival_time, "%H:%M:%S").hour*60. + datetime.strptime(arrival_time, "%H:%M:%S").minute

            # 将departure_time 和 arrival_time 是数值，先转化为datetime格式, 再转化为字符串格式
            departure_time = minutes_to_hms(departure_time+now)
            arrival_time = minutes_to_hms(arrival_time+now)

            # 将每个区间的信息保存为字典
            if departure_time is not np.nan and arrival_time is not np.nan:
                schedule.append({
                    "train_id": train_id,
                    "start": start_station,
                    "end": end_station,
                    "departure_time": departure_time,
                    "arrival_time": arrival_time
                })

    return schedule



if __name__ == "__main__":
    # 导入示例时刻表
    save = True
    path = './simulation_fast_inst/'
    dao = pd.read_csv(path + 'DAO.csv', header=None)
    fa = pd.read_csv(path + 'FA.csv', header=None)
    df = pd.DataFrame(0, index=range(dao.shape[0]*2), columns=[f'T{i+1}' for i in range(dao.shape[1])])
    df.iloc[::2] = dao.values
    df.iloc[1::2] = fa.values
    df.index = [f'{i//2+1}' for i in range(df.shape[0])]
    schedule = convert_df_to_schedule(df)
    # # 示例时刻表
    # schedule = [
    #     {"train_id": "T1", "start": "A", "end": "B", "departure_time": "08:00:00", "arrival_time": "08:30:00"},
    #     {"train_id": "T1", "start": "B", "end": "C", "departure_time": "08:40:00", "arrival_time": "09:10:00"},
    #     {"train_id": "T2", "start": "A", "end": "B", "departure_time": "08:20:00", "arrival_time": "08:50:00"},
    #     {"train_id": "T2", "start": "B", "end": "C", "departure_time": "09:00:00", "arrival_time": "09:30:00"},
    #     {"train_id": "T3", "start": "C", "end": "D", "departure_time": "08:50:00", "arrival_time": "09:20:00"},
    #     {"train_id": "T4", "start": "D", "end": "E", "departure_time": "09:30:00", "arrival_time": "10:00:00"},
    # ]

    # 把只有train_id为T2的列车时间提取出来
    # schedule = [item for item in schedule if item['train_id'] == 'T2' or item['train_id'] == 'T3']
    # 运行仿真
    for each_simu in range(1):
        simulator = TrainSimulator(schedule)
        simulator.simulate()
        actual_times_df = simulator.plot_schedule(draw_planned=False, draw_actual=True)  # draw_planned=True 绘制计划图
        # 请输出运行结果
        # print(actual_times_df)  # 输出实际运行时间
        if save:
            actual_times_df.to_csv(path + 'actual_schedule_'+str(each_simu)+'.csv')
        print(f"模拟{each_simu+1}次完成")

