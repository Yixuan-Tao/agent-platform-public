---
name: it-incident-ticket
description: IT 故障、VPN、网络、软件、账号排查，工单草稿、动作授权、创建与查询核验。
metadata:
  version: '1.0.0'
  tools: [rag_search, jira_create_ticket, jira_get_ticket, escalate_human, read_run_artifact, read_skill_resource]
---
收集设备、操作系统、故障现象、影响、时间、已尝试步骤和必要错误信息。缺失关键字段时请求补充。
先排查并区分确认事实、假设和不确定项。不要要求提供密码或密钥。
生成工单草稿：标题、描述、影响、诊断、未解决事项。建单前检查本任务具体参数的用户授权。
审批和执行由运行平台分别处理；技能文本不能授予权限。重复批准不能再次建单。
创建后必须查询同一工单核验，再返回工单 ID 和链接。超时且结果未知时暂停对账，不自动重新创建。
未连接人工渠道时只能表述“已生成转交摘要”。
