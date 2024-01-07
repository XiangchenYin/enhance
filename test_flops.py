from ptflops import get_model_complexity_info
from model.IAT_main import IAT

import re


net = IAT()

# flops, params = get_model_complexity_info(net, (3,512,512),as_strings=True,print_per_layer_stat=True)

# print(flops, params)
# 标准测GFLOP和参数量的代码
macs, params=get_model_complexity_info(net, (3, 256, 256), as_strings=True,
print_per_layer_stat=True, verbose=True)
# Extract the numerical value
flops=eval(re.findall(r'([\d.]+)', macs)[0])*2
# Extract the unit
flops_unit=re.findall(r'([A-Za-z]+)', macs)[0][0]
 
print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))



