categories = [
    'OE_TO_maths_en_COMP', # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_COMP', # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_CEE', # OpenEnded - TextOnly - maths - CEE
    'OE_TO_physics_en_COMP', # OpenEnded - TextOnly - physics - COMP
    'OE_TO_physics_zh_CEE' # OpenEnded - TextOnly - physics - CEE
]

_olympiad_maths_weights = {'OE_TO_maths_en_COMP': 674, 'OE_TO_maths_zh_COMP': 408, 'OE_TO_maths_zh_CEE': 1240}
_olympiad_maths_weights = {'OlympiadBench_' + k : v for k,v in _olympiad_maths_weights.items()}

_olympiad_physics_weights = {'OE_TO_physics_en_COMP': 236, 'OE_TO_physics_zh_CEE': 115}
_olympiad_physics_weights = {'OlympiadBench_' + k : v for k,v in _olympiad_physics_weights.items()}

_olympiad_weighted = {**_olympiad_maths_weights, **_olympiad_physics_weights}

OlympiadBench_summary_groups = [
    {'name': 'OlympiadBench-maths', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in ['OE_TO_maths_en_COMP', 'OE_TO_maths_zh_COMP', 'OE_TO_maths_zh_CEE']]},
    {'name': 'OlympiadBench-maths-weighted', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in ['OE_TO_maths_en_COMP', 'OE_TO_maths_zh_COMP', 'OE_TO_maths_zh_CEE']], 'weights': _olympiad_maths_weights},
    
    {'name': 'OlympiadBench-physics', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in ['OE_TO_physics_en_COMP', 'OE_TO_physics_zh_CEE']]},
    {'name': 'OlympiadBench-physics-weighted', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in ['OE_TO_physics_en_COMP', 'OE_TO_physics_zh_CEE']], 'weights': _olympiad_physics_weights},
    
    {'name': 'OlympiadBench', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in categories]},
    {'name': 'OlympiadBench-weighted', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in categories], 'weights': _olympiad_weighted},
]

math_categories = [
    'OE_TO_maths_en_COMP', # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_COMP', # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_CEE', # OpenEnded - TextOnly - maths - CEE
]

physics_categories = [
    'OE_TO_physics_en_COMP', # OpenEnded - TextOnly - physics - COMP
    'OE_TO_physics_zh_CEE' # OpenEnded - TextOnly - physics - CEE
]


OlympiadBenchMath_summary_groups = [
    {'name': 'OlympiadBenchMath', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in math_categories]},
]


OlympiadBenchPhysics_summary_groups = [
    {'name': 'OlympiadBenchPhysics', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in physics_categories]},
]
