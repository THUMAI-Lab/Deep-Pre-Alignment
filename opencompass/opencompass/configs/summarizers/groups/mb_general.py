# | dataset | version | metric | mode | Qwen3-8B-qwen3-nothink-sft-vllm |                           
# | openai_mmmlu_lite_AR-XY | 07891e | accuracy | gen | 26.67 |  
# | openai_mmmlu_lite_BN-BD | 0c33f9 | accuracy | gen | 53.61 |                     
# | openai_mmmlu_lite_DE-DE | 2f4124 | accuracy | gen | 37.89 |                     
# | openai_mmmlu_lite_ES-LA | 555bbc | accuracy | gen | 68.35 |                     
# | openai_mmmlu_lite_FR-FR | 97f4e3 | accuracy | gen | 69.40 |                     
# | openai_mmmlu_lite_HI-IN | 94cd28 | accuracy | gen | 60.84 |                     
# | openai_mmmlu_lite_ID-ID | b65293 | accuracy | gen | 68.84 |                     
# | openai_mmmlu_lite_IT-IT | bf8b68 | accuracy | gen | 68.70 |                     
# | openai_mmmlu_lite_JA-JP | 45635b | accuracy | gen | 67.79 |                     
# | openai_mmmlu_lite_KO-KR | bd4d2a | accuracy | gen | 65.40 |                          
# | openai_mmmlu_lite_PT-BR | 0b103d | accuracy | gen | 52.00 |                          
# | openai_mmmlu_lite_SW-KE | 75318c | accuracy | gen | 44.21 |                          
# | openai_mmmlu_lite_YO-NG | 75318c | accuracy | gen | 36.35 |                          
# | openai_mmmlu_lite_ZH-CN | 14e7b8 | accuracy | gen | 74.25 | 
# openai_mmmlu_lite
openai_mmmlu_lite_summary_groups = []

openai_mmmlu_lite_summary_groups.append(
    {'name': 'openai_mmmlu_lite', 'subsets': [f'openai_mmmlu_lite_{c}' for c in ['AR-XY', 'BN-BD', 'DE-DE', 'ES-LA', 'FR-FR', 'HI-IN', 'ID-ID', 'IT-IT', 'JA-JP', 'KO-KR', 'PT-BR', 'SW-KE', 'YO-NG', 'ZH-CN']]})
