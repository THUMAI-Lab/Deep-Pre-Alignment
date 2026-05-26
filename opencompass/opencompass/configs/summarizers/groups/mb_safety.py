mb_safety_summary_groups = []

_mb_safety_all = [
    '违反社会主义核心价值观',
    '歧视性内容',
    '侵犯他人合法权益',
    '商业违法违规',
    '无法满足特定服务类型的安全需求',
]

mb_safety_summary_groups.append(
    {'name': 'mb_safety', 'subsets': [f'mb_safety_{c}' for c in _mb_safety_all]})
