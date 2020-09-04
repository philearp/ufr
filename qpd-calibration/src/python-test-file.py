import numpy as np
from datetime import datetime

output_filename = 'outputfile.txt'

# datetime object containing current date and time
now = datetime.now()
 
print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
date_txt = "date and time =" + dt_string
print(date_txt)	

print(type(date_txt))

text_file = open(output_filename, "w")
n = text_file.write(date_txt)
text_file.close()