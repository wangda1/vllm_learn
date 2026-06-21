#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
course19 · Demo 4 —— Proxy 如何编排 P/D，xPyD 如何路由（6 步流程 + 服务发现）
=============================================================================

对应 DOC.md：第一章 1.2「PD 分离架构概述（6 步流程）」、
            第三章 3.1「执行链路概述」、第五章 5.3「xPyD 架构」

前三课讲了「KV 怎么算、怎么搬、怎么对号」。这一课讲 **谁来指挥**：
API Proxy 是整个 PD 分离应用的「总调度」，它本身不带 GPU、纯 CPU，
负责把一个请求拆成「先 P 后 D」两段，并在 xPyD（x 个 Prefill + y 个 Decode）里选路。

本 demo 用纯 Python 对象模拟整条链路（不开网络），完整复刻 vLLM 官方
disagg_proxy_p2p_nccl_xpyd.py 的真实逻辑，运行：

    python proxy_xpyd_demo.py

要讲透的两个核心设计：
  ① 6 步流程：Proxy 把 max_tokens 改成 1 发给 P → 丢弃 P 的响应 → 把原始请求发给 D。
  ② request_id 编址：Proxy 生成的 request_id 里【内嵌了 P 和 D 的地址】，
     P 算完 KV 后正是靠解析这个 request_id 才知道该把 KV 发给哪个 D。
"""

import itertools


# --------------------------------------------------------------------------
# 1. 模拟一个 vLLM 实例（P 或 D）。真实里它们是独立进程 / 独立 GPU。
# --------------------------------------------------------------------------
class VllmInstance:
    def __init__(self, role, http_addr, zmq_addr):
        self.role = role              # "prefill" 或 "decode"
        self.http_addr = http_addr    # HTTP 服务地址（Proxy 转发请求用）
        self.zmq_addr = zmq_addr      # ZMQ 地址（P↔D 之间传 KV 元数据/握手用）
        # D 节点的「KV 信箱」：request_id -> 收到的 KV（模拟后台接收线程的 recv_store）
        self.kv_inbox = {}

    # ---- P 节点：只做 prefill，max_tokens 已被 Proxy 改成 1 ----
    def handle_prefill(self, req):
        assert self.role == "prefill"
        rid = req["request_id"]
        # P 解析 request_id，得知该把 KV 发给哪个 D（这就是 parse_request_id 的作用）
        decode_zmq = parse_decode_addr(rid)
        kv = f"KV(prompt={req['prompt']!r})"            # 模拟 prefill 产出的 KV cache
        # 主动把 KV 推送给目标 D（PUT_ASYNC：NCCL 点对点直发）
        DECODE_BY_ZMQ[decode_zmq].kv_inbox[rid] = kv
        print(f"      [P {self.http_addr}] prefill 完成(只生成 1 token)，"
              f"按 request_id 把 KV 直发给 D(zmq={decode_zmq})")
        # P 也会返回一个响应，但里面只有 1 个 token —— Proxy 会丢弃它
        return {"text": "<first_token>", "note": "max_tokens=1, 将被 Proxy 丢弃"}

    # ---- D 节点：接收 KV，跳过 prefill，直接 decode 到 max_tokens ----
    def handle_decode(self, req):
        assert self.role == "decode"
        rid = req["request_id"]
        kv = self.kv_inbox.pop(rid, None)
        if kv is None:
            return {"error": "KV 未送达，D 只能回退到自己重新 prefill（TTFT 飙升）"}
        print(f"      [D {self.http_addr}] 命中 KV：{kv} → 跳过 prefill，"
              f"直接 decode 到 max_tokens={req['max_tokens']}")
        toks = " ".join(f"w{i}" for i in range(req["max_tokens"]))
        return {"text": f"{req['prompt']} -> [{toks}]"}


# --------------------------------------------------------------------------
# 2. request_id 编址：把 P/D 地址编进 request_id，再解析出来
#    真实格式： ___prefill_addr_{P_zmq}___decode_addr_{D_zmq}_{uuid}
# --------------------------------------------------------------------------
_uuid_counter = itertools.count(1)


def make_request_id(prefill_zmq, decode_zmq):
    return f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_uuid{next(_uuid_counter)}"


def parse_decode_addr(request_id):
    """从 request_id 里解析出 D 的 zmq 地址（对应 connector 的 parse_request_id）。"""
    return request_id.split("___decode_addr_")[1].rsplit("_uuid", 1)[0]


# --------------------------------------------------------------------------
# 3. API Proxy：纯 CPU 的总调度。这是整个 demo 的主角。
# --------------------------------------------------------------------------
class ApiProxy:
    def __init__(self, prefill_instances, decode_instances):
        self.P = prefill_instances     # x 个 Prefill
        self.D = decode_instances      # y 个 Decode
        self.count = 0                 # 轮询计数器（round-robin）

    def handle(self, client_request):
        """完整复刻官方 6 步流程。client_request = {"prompt":..., "max_tokens":...}"""
        # —— 选路：轮询选一个 P 和一个 D（count % len，真实代码就是这样）——
        p = self.P[self.count % len(self.P)]
        d = self.D[self.count % len(self.D)]
        self.count += 1

        # 生成内嵌 P/D 地址的 request_id
        rid = make_request_id(p.zmq_addr, d.zmq_addr)
        print(f"\n  Proxy 选路: 客户端请求 👉 [P {p.http_addr}] 👉 [D {d.http_addr}]  "
              f"(count={self.count - 1})")
        print(f"  生成 request_id = {rid}")

        # 【步骤1】把请求复制一份，max_tokens 强制改成 1，发给 P
        prefill_req = dict(client_request)
        prefill_req["max_tokens"] = 1
        prefill_req["request_id"] = rid
        # 【步骤2】P 完成 prefill，生成 KV 并主动发给 D（见 handle_prefill）
        p_resp = p.handle_prefill(prefill_req)
        # 【步骤3】Proxy 主动丢弃 P 的响应（只是为了触发 prefill，不要它的 token）
        print(f"      Proxy 丢弃 P 的响应: {p_resp['note']}")

        # 【步骤4】把【原始】请求（完整 max_tokens）发给 D
        decode_req = dict(client_request)
        decode_req["request_id"] = rid
        # 【步骤5】D 读取共享 KV，继续 decode
        d_resp = d.handle_decode(decode_req)
        # 【步骤6】把 D 的结果返回客户端
        print(f"      Proxy 把 D 的最终结果返回客户端")
        return d_resp


# --------------------------------------------------------------------------
# 4. 服务发现：P/D 启动后向 Proxy 注册自己（真实里通过 ZMQ register 消息）
# --------------------------------------------------------------------------
DECODE_BY_ZMQ = {}   # 全局：zmq_addr -> DecodeInstance，供 P 直发 KV 时查找


def build_cluster(num_p, num_d):
    """搭一个 xPyD 集群：num_p 个 Prefill + num_d 个 Decode。"""
    P, D = [], []
    for i in range(num_p):
        P.append(VllmInstance("prefill", f"10.0.0.{10+i}:2000{i}", f"10.0.0.{10+i}:2100{i}"))
    for j in range(num_d):
        d = VllmInstance("decode", f"10.0.0.{20+j}:2000{j}", f"10.0.0.{20+j}:2200{j}")
        D.append(d)
        DECODE_BY_ZMQ[d.zmq_addr] = d   # 注册到服务发现表
    return P, D


# ==========================================================================
# main
# ==========================================================================
def demo(num_p, num_d, num_requests):
    print("=" * 70)
    print(f"部署形态: {num_p}P{num_d}D  （{num_p} 个 Prefill 实例 + {num_d} 个 Decode 实例）")
    print("=" * 70)
    DECODE_BY_ZMQ.clear()
    P, D = build_cluster(num_p, num_d)
    proxy = ApiProxy(P, D)

    print(f"集群已注册: P={[p.http_addr for p in P]}")
    print(f"           D={[d.http_addr for d in D]}")

    for n in range(num_requests):
        req = {"prompt": f"问题{n}", "max_tokens": 4}
        resp = proxy.handle(req)
        print(f"  ✅ 客户端收到: {resp.get('text', resp)}")


if __name__ == "__main__":
    print(__doc__)

    # 场景一：最小 1P1D —— 看清 6 步流程
    demo(num_p=1, num_d=1, num_requests=2)

    # 场景二：1P3D —— 看清「同一个 P 喂多个 D」，请求被轮询分散到不同 D
    print("\n")
    demo(num_p=1, num_d=3, num_requests=4)

    # 场景三：3P1D —— 长文本场景，多个 P 并行 prefill，KV 都汇到 1 个 D
    print("\n")
    demo(num_p=3, num_d=1, num_requests=3)

    print("""
======================================================================
本课要点回顾
======================================================================
  · API Proxy 纯 CPU，是 PD 分离应用的「总调度」，对客户端表现为一个普通 OpenAI 接口。
  · 6 步流程的精髓：max_tokens=1 把 P「骗」成只做 prefill；丢弃 P 响应；原始请求给 D。
  · request_id 内嵌了 P/D 地址 —— 这是「去中心化路由」的关键：
    P 不需要 Proxy 再告诉它发给谁，自己解析 request_id 就知道目标 D 的地址。
  · xPyD 的扩缩容只是改变 Proxy 轮询的列表长度；P↔D 是点对点 NCCL，
    新增/删除实例无需重启整个系统（动态扩展能力）。

至此 course19 四个 demo 跑完：
  why → handoff → block_remap → proxy_xpyd，
  从「为什么分」到「KV 怎么交接、怎么对号、谁来指挥」完整闭环。
  更细的源码走读见 DOC.md。
""")
