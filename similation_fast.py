import heapq
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
import copy

class TrainSimulator:
    def __init__(self, schedule):
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

        # 初始化事件队列
        for item in schedule:
            train_id = item["train_id"]
            dep_time = datetime.strptime(item["departure_time"], "%H:%M:%S")
            arr_time = datetime.strptime(item["arrival_time"], "%H:%M:%S")
            heapq.heappush(self.events, (dep_time, "depart", train_id, item))
            # 如果self.trains中不存在train_id，则添加一个新的列车状态
            if train_id not in self.trains:
                self.trains[train_id] = {
                    "current_section": None,
                    "next_departure_time": dep_time,
                    "next_arrival_time": arr_time,
                    "status": "waiting",
                }

    def simulate(self):
        """
        开始仿真
        """
        while self.events:
            # 取出最早的事件
            event_time, event_type, train_id, event_data = heapq.heappop(self.events)
            self.current_time = event_time

            # 当运行到这个列车的事件后，所有后续列车全部都删除
            self.events = [ev for ev in self.events if not ev[2] == train_id]

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
                    if item[2] == log["train_id"] and item[1] == "arrive" and item[3]["end"] == section[1]:
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
        delay = random.randint(-5, 10)  # 随机延误，单位分钟
        actual_arrival_time = planned_departure_time + planned_operation_time + timedelta(minutes=delay)  # 按照计划来跑，不存在最短运行时间，赶点的情况

        # 检查到站间隔约束（与前一列车至少相隔 3 分钟）
        for log in reversed(self.logs):
            if log["event"] == "arrive" and log["station"] == section[1]:
                last_arrival_time = log["time"]
                if actual_arrival_time < last_arrival_time + timedelta(minutes=3):
                    actual_arrival_time = last_arrival_time + timedelta(minutes=3)
                break

        # 保证列车下一站的到站时间大于计划时间，
        if actual_arrival_time < planned_headway:
            actual_arrival_time = planned_headway

        # 添加到站事件
        heapq.heappush(self.events, (actual_arrival_time, "arrive", train_id, event_data))
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
        next_departure_time = self.current_time + timedelta(minutes=2)

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

                planed_next_arrival = planed_next_operation_time + planned_next_departure
                # 保证新生成的事件的到站时间满足越行约束
                for events in reversed(self.events):
                    if events[1] == "depart" and events[3]["end"] == next_end and events[2] != train_id:
                        if events[0] < planned_next_departure:
                            if planed_next_arrival < datetime.strptime(events[3]["arrival_time"], "%H:%M:%S") + timedelta(minutes=3):
                                planed_next_arrival = events[3]["arrival_time"] + timedelta(minutes=3)
                            break

                # 更新事件队列中相关的发车事件
                new_event = (planned_next_departure, "depart", train_id, {
                    "start": next_start,
                    "end": next_end,
                    "departure_time": planned_next_departure.strftime("%H:%M:%S"),
                    "arrival_time": planed_next_arrival.strftime("%H:%M:%S"),
                })

                # 过滤掉旧的发车事件并插入新的事件
                self.events = [
                    ev for ev in self.events if not (ev[1] == "depart" and ev[2] == train_id)
                ]
                heapq.heappush(self.events, new_event)

                break

    def plot_schedule(self, draw_planned=True):
        """
        绘制列车运行时刻表，包括计划图和实际图，同时标记延误信息。
        同一列车的计划图和实际图使用一致的颜色。
        """
        plt.figure(figsize=(14, 7))
        stations = list(set([event["station"] for event in self.logs]))
        stations = np.sort([int(stations[i]) for i in range(len(stations))])
        stations = [str(i) for i in stations]
        train_ids = list(set([event["train_id"] for event in self.logs]))

        # 为每辆列车分配固定颜色
        colors = list(mcolors.TABLEAU_COLORS.values())
        train_colors = {train_id: colors[i % len(colors)] for i, train_id in enumerate(train_ids)}

        for train_id in train_ids:
            # 实际运行数据
            actual_times = []
            actual_positions = []
            delays = []  # 延误时间记录

            # 计划运行数据
            planned_times = []
            planned_positions = []

            for event in self.logs:
                if event["train_id"] == train_id:
                    actual_times.append(event["time"])
                    actual_positions.append(stations.index(event["station"]))

                    # 计算延误时间（实际时间与计划时间的差值，单位分钟）
                    planned_time = datetime.strptime(event["planned_time"], "%H:%M:%S")
                    delay = (event["time"] - planned_time).total_seconds() / 60
                    delays.append(delay)

                    planned_times.append(planned_time)
                    planned_positions.append(stations.index(event["station"]))

            # 获取该列车的颜色
            train_color = train_colors[train_id]

            # 绘制计划图（虚线）
            if draw_planned:
                plt.plot(planned_times, planned_positions, linestyle='--', color=train_color, label=f"{train_id} - plan")

            # 绘制实际图（实线）
            plt.plot(actual_times, actual_positions, marker='o', color=train_color, label=f"{train_id} - actual")

            # 标记延误信息
            for t, p, d in zip(actual_times, actual_positions, delays):
                if d > 0:
                    plt.text(t, p, f"+{int(d)}'", color="red", fontsize=9, ha='left', va='bottom')
                elif d < 0:
                    plt.text(t, p, f"{int(d)}'", color="blue", fontsize=9, ha='left', va='bottom')

        # 格式化横坐标时间
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # 10分钟间隔

        plt.yticks(range(len(stations)), stations)
        plt.xlabel("Time")
        plt.ylabel("Station")
        plt.title("Train schedule (plan vs actual)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


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
    time_delta = timedelta(minutes=minutes.astype(float))

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

            # 将departure_time 和 arrival_time 是数值，先转化为datetime格式, 再转化为字符串格式
            departure_time = minutes_to_hms(departure_time+now)
            arrival_time = minutes_to_hms(arrival_time+now)

            # 将每个区间的信息保存为字典
            schedule.append({
                "train_id": train_id,
                "start": start_station,
                "end": end_station,
                "departure_time": departure_time,
                "arrival_time": arrival_time
            })

    return schedule


import pandas as pd
# 导入示例时刻表
path = './simulation_fast_inst/'
dao = pd.read_csv(path + 'DAO.csv')
fa = pd.read_csv(path + 'FA.csv')
df = pd.DataFrame(0, index=range(dao.shape[0]*2), columns=[f'T{i+1}' for i in range(dao.shape[1])])
df.iloc[::2] = dao.values
df.iloc[1::2] = fa.values
df.index = [f'{i//2+1}' if i % 2 == 0 else f'{i//2+1}' for i in range(df.shape[0])]
schedule = convert_df_to_schedule(df)
# # 示例时刻表（更复杂）
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
simulator = TrainSimulator(schedule)
simulator.simulate()
simulator.plot_schedule(draw_planned=False)