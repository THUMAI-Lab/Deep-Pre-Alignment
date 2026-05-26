financeIQ_subjects = [
    '注册会计师（CPA）',
    '银行从业资格',
    '证券从业资格',
    '基金从业资格',
    '保险从业资格CICE',
    '经济师',
    '税务师',
    '期货从业资格',
    '理财规划师',
    '精算师-金融数学',
]

FinanceIQ_summary_groups = []

FinanceIQ_summary_groups.append({'name': 'FinanceIQ', 'subsets': [f'FinanceIQ-{financeIQ_subject}' for financeIQ_subject in financeIQ_subjects]})

FinanceIQ_0shot_summary_groups = []
FinanceIQ_0shot_summary_groups.append({'name': 'FinanceIQ-0shot', 'subsets': [f'FinanceIQ-{financeIQ_subject}-0shot' for financeIQ_subject in financeIQ_subjects]})
