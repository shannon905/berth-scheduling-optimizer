"""
泊位调度优化程序
使用 PuLP 建立整数规划模型，最小化所有船舶的总等待时间，
并通过 Flask 提供 HTTP API 接口。
"""

from flask import Flask, request, jsonify
import pulp

app = Flask(__name__)


def optimize_berth_scheduling(ships, num_berths):
    """
    使用 PuLP 建立整数规划模型求解泊位调度问题。

    参数:
        ships: 船舶列表，每艘船为 dict，包含 name, arrival, service
        num_berths: 可用泊位数量

    返回:
        dict: 包含每艘船的分配结果及总等待时间
    """
    n = len(ships)
    if n == 0:
        return {"ships": [], "total_wait_time": 0}

    arrivals = [s["arrival"] for s in ships]
    services = [s["service"] for s in ships]
    names = [s["name"] for s in ships]

    # big-M 的取值：足够大以放松非绑定约束
    M = max(arrivals) + sum(services) + 1

    # 创建整数规划问题，目标为最小化总等待时间
    prob = pulp.LpProblem("Berth_Scheduling", pulp.LpMinimize)

    # ====== 决策变量 ======

    # x[i][b]: 二进制变量，船 i 是否分配到泊位 b
    x = [[pulp.LpVariable(f"x_{i}_{b}", cat="Binary") for b in range(num_berths)]
         for i in range(n)]

    # s[i]: 整数变量，船 i 的开始服务时间
    s = [pulp.LpVariable(f"s_{i}", lowBound=0, cat="Integer") for i in range(n)]

    # y[i][j]: 二进制变量，船 i 是否在船 j 之前开始服务
    # 用于 big-M 方法处理同一泊位上的服务不重叠约束
    y = [[pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(n)]
         for i in range(n)]

    # ====== 目标函数：最小化总等待时间 ======
    wait_time = [s[i] - arrivals[i] for i in range(n)]
    prob += pulp.lpSum(wait_time)

    # ====== 约束条件 ======

    # 约束 1：每艘船必须分配且仅分配一个泊位
    for i in range(n):
        prob += pulp.lpSum(x[i][b] for b in range(num_berths)) == 1, f"assign_berth_{i}"

    # 约束 2：开始服务时间不能早于到达时间
    for i in range(n):
        prob += s[i] >= arrivals[i], f"no_early_start_{i}"

    # 约束 3：同一泊位上任意两艘船的服务时间不能重叠（big-M 方法）
    # 对每对船 (i, j) 其中 i < j，以及每个泊位 b：
    #   若 i 和 j 都在泊位 b，且 i 在 j 之前服务 => s[i] + svc[i] <= s[j]
    #   若 i 和 j 都在泊位 b，且 j 在 i 之前服务 => s[j] + svc[j] <= s[i]
    # 使用 big-M 放松非同一泊位或相对顺序相反时的约束
    for i in range(n):
        for j in range(i + 1, n):
            for b in range(num_berths):
                # i 先于 j 服务，或 i、j 不在同一泊位 b 时约束放松
                prob += (s[i] + services[i] <= s[j]
                         + M * (1 - y[i][j])
                         + M * (2 - x[i][b] - x[j][b])), \
                    f"overlap_{i}_{j}_{b}_a"
                # j 先于 i 服务，或 i、j 不在同一泊位 b 时约束放松
                prob += (s[j] + services[j] <= s[i]
                         + M * y[i][j]
                         + M * (2 - x[i][b] - x[j][b])), \
                    f"overlap_{i}_{j}_{b}_b"

    # ====== 求解 ======
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return {"error": f"No optimal solution found: {pulp.LpStatus[prob.status]}"}

    # ====== 提取结果 ======
    result_ships = []
    total_wait = 0

    for i in range(n):
        start_time = int(pulp.value(s[i]))
        end_time = start_time + services[i]
        wait = start_time - arrivals[i]
        total_wait += wait

        # 找出船 i 被分配到哪个泊位
        berth_assigned = None
        for b in range(num_berths):
            if pulp.value(x[i][b]) > 0.5:
                berth_assigned = b
                break

        result_ships.append({
            "name": names[i],
            "arrival": arrivals[i],
            "service": services[i],
            "berth": berth_assigned,
            "start_time": start_time,
            "end_time": end_time,
            "wait_time": wait
        })

    return {
        "ships": result_ships,
        "total_wait_time": total_wait
    }


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    HTTP API：接收 JSON 输入，返回泊位调度优化结果。
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    ships = data.get("ships", [])
    berths = data.get("berths", 1)

    if not ships:
        return jsonify({"error": "At least one ship is required"}), 400

    result = optimize_berth_scheduling(ships, berths)
    return jsonify(result)


if __name__ == "__main__":
    # ====== 本地测试 ======
    print("=" * 50)
    print("本地测试：3 艘船，2 个泊位")
    print("=" * 50)

    test_ships = [
        {"name": "A", "arrival": 1, "service": 2},
        {"name": "B", "arrival": 2, "service": 1},
        {"name": "C", "arrival": 3, "service": 3}
    ]
    test_berths = 2

    result = optimize_berth_scheduling(test_ships, test_berths)

    if "error" in result:
        print(f"求解失败: {result['error']}")
    else:
        for s in result["ships"]:
            print(f"  船舶 {s['name']}: 泊位 {s['berth']}, "
                  f"到达 {s['arrival']}, 开始 {s['start_time']}, "
                  f"结束 {s['end_time']}, 等待 {s['wait_time']}")
        print(f"  总等待时间: {result['total_wait_time']}")

    # ====== 启动 Flask 服务 ======
    print("\nFlask 服务启动在 http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
