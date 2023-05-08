import numpy as np
import pandas as pd
import re
import streamlit as st

class Raw_data:
    def init(self):
        pass

    def read_param(self):
        target_words = ('F1LEFT', 'F1RIGHT', 'F2LEFT', 'F2RIGHT','NROWS', 'NCOLS')
        # Open the text file in read mode

        data = []
        with open(uploaded_file, 'r') as file:
            lines = file.readlines()

            # Iterate through the lines and read until a line is encountered
            # that does not begin with a number
            for line in lines:
                # Strip leading and trailing whitespace from the line
                stripped_line = line.strip()

                # Check if the line begins with a number
                if stripped_line and stripped_line[0].isdigit():
                    # Break the loop when a line is encountered that does not begin with a number
                # Print the line or perform any desired operations
                    break

                #print(line)

                for target_word in target_words:
                    if target_word in line:
                        data.append(line)

        f1left = 0
        f1right = 0
        f2left = 0
        f2right = 0
        nrows = 0
        ncols = 0

        for line in data:  
            if 'F1LEFT' in line:
                f1left = float(re.findall(r'\d+\.\d+', line)[0])
                f1right = float(re.findall(r'\d+\.\d+', line)[1])
            elif 'F2LEFT' in line:
                f2left = float(re.findall(r'\d+\.\d+', line)[0])
                f2right = float(re.findall(r'\d+\.\d+', line)[1])
            elif 'NROWS' in line:
                nrows = int(re.findall(r'\d+', line)[0])
            elif 'NCOLS' in line:
                ncols = int(re.findall(r'\d+', line)[0])

        print('F1LEFT: '+str(f1left))
        print('F1RIGHT: '+str(f1right))
        print('F2LEFT: '+str(f2left))
        print('F2RIGHT: '+str(f2right))
        print('NROWS: '+str(nrows))
        print('NCOLS: '+str(ncols))

        df = np.loadtxt(datatxt)
        df = np.reshape(df, (nrows, -1))
        col_names = np.linspace(f2left, f2right, ncols)
        row_names = np.linspace(f1left, f1right, nrows)
        #np.savetxt('data.txt', df, delimiter = ',')

        np.savetxt('tmp/f2.txt', col_names, fmt = '%.3f')
        np.savetxt('tmp/f1.txt', row_names, fmt = '%.3f')
        df2 = pd.DataFrame(df)
        df2.to_csv('data2d.txt', index=False)

#Raw_data.read_param('exam2d_CH.txt')