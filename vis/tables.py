import numpy as np

avg = '80.0	0.97	87.9	0.85	73.0	0.89	45.8	3.17	99.2	0.63	77.2	1.63'
sd = ''

avg_tokens = avg.split('	')
sd_tokens = sd.split('	')
str = ''
if sd != '':
    if len(avg_tokens) - 2 == len(sd_tokens):
        for i in np.arange(0, len(avg_tokens)):
            if i < 10:
                str = str + '&' + avg_tokens[i] + ' $\pm$ ' + sd_tokens[i] + ' '
            else:
                str = str + '&' + avg_tokens[i] + ' '
else:
    for i in np.arange(0, len(avg_tokens)):
        str = str + '&' + avg_tokens[i] + ' '

print(str)
