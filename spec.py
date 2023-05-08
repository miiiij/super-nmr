import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

st.header('NMR Data Processing Site for NMR lab')
uploaded_files = st.file_uploader("Choose a data file in csv or txt format", type=['csv','txt'],accept_multiple_files=True)

def opt_1d_spec():
## read data ##
    col_name = ['Index', 'Intensity', 'Hz', 'ppm']
    df = pd.read_csv(uploaded_file, names=col_name)
    df = df.drop(labels='Index', axis=1)

    hide_df = st.checkbox("Show DataFrame", key = key2)
    if hide_df:  
     st.dataframe(df)

    with st.sidebar:
      ## title ##
      if i + 1 == 1:
        st.header(str(i + 1)+'st file')
      elif i + 1 == 2:
        st.header(str(i + 1)+'nd file')
      elif i + 1 == 3:
        st.header(str(i + 1),'rd file')
      else:
        st.header(str(i + 1),'th file')

       ## draw curves ##
      st.write('Range for x(ppm): ', df['ppm'].min(), '~', df['ppm'].max())
      st.write('\n Thumbnail spectrum is')

      fig, ax = plt.subplots(figsize = (12,6))
      X = df['ppm']
      Y = df['Intensity']
      ax.set_xlabel('Chemical shift, ppm')
      ax.set_ylabel('Intensity, -')
      ax.plot(X, Y)
      ax.invert_xaxis()
      fig
      

      #st.write('Range for x(ppm): ', x2, '~', x1)
    st.write('Set the range for x-axis >_<')
    x = st.sidebar.slider('x range',
                  float(df['ppm'].min()), float(df['ppm'].max()),
                    (float(df['ppm'].min()), float(df['ppm'].max())), key = key)
    x1 = x[0]
    x2 = x[1]

    target_value_1 = x1
    nearest_idx_1 = (df['ppm'] - target_value_1).abs().idxmin()
    #nearest_int_1 = df.loc[nearest_idx_1]['Intensity']
    target_value_2 = x2
    nearest_idx_2 = (df['ppm'] - target_value_2).abs().idxmin()
    #nearest_int_2 = df.loc[nearest_idx_2]['Intensity']

    fig, ax = plt.subplots(figsize = (8,4))
    Xcrop = df['ppm'][nearest_idx_2:nearest_idx_1]
    Ycrop = df['Intensity'][nearest_idx_2:nearest_idx_1]
    ax.set_xlabel('Chemical shift, ppm')
    ax.set_ylabel('Intensity, -')
    ax.plot(Xcrop, Ycrop, lw =1)
    ax.invert_xaxis()
    fig


def read_param():
  target_words = ('F1LEFT', 'F1RIGHT', 'F2LEFT', 'F2RIGHT','NROWS', 'NCOLS')

  data = []
  df = pd.read_csv(uploaded_file)
  df.to_csv('./tmp/pizza.csv', sep = ' ', index = False)
  
  with open('./tmp/pizza.txt', 'r') as file:
    lines = file.readlines()

    for line in lines:
        stripped_line = line.strip()

        if stripped_line and stripped_line[0].isdigit():
            break

        print(line)

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

  df = np.loadtxt('./tmp/pizza.txt',
                  comments = ['"', '#'])
  df = np.reshape(df,(nrows, -1))
  col_names = np.linspace(f2left, f2right, ncols)
  row_names = np.linspace(f1left, f1right, nrows)

  np.savetxt('tmp/f2.txt', col_names, fmt = '%.3f')
  np.savetxt('tmp/f1.txt', row_names, fmt = '%.3f')
  df2 = pd.DataFrame(df)
  df2.to_csv('./tmp/data2d.txt', index=False)

def opt_2d_spec():
  '''if not os.path.exists('./tmp/data2d.txt'):
      print("Make a 2d data file.")
      read_param()'''
  
  df = pd.read_csv('./tmp/data2d.txt')
  df

  hide_df = st.checkbox("Show DataFrame", key = key2)
  if hide_df:
    st.dataframe(df.set_index(df.columns[0]))

  with st.sidebar:
    ## title ##
    if i + 1 == 1:
      st.header(str(i + 1)+'st file')
    elif i + 1 == 2:
      st.header(str(i + 1)+'nd file')
    elif i + 1 == 3:
      st.header(str(i + 1),'rd file')
    else:
      st.header(str(i + 1),'th file')

      ## draw curves ##

    x = np.loadtxt('./tmp/f2.txt')
    y = np.loadtxt('./tmp/f1.txt')
    f2m = float(min(x))
    f2M = float(max(x))
    f1m = float(min(y))
    f1M = float(max(y))

     # 'Range %.2f ~ %.2f ppm' % (float(df.columns[1]), float(df.columns[-1]))
    st.write('Range for F2: %.2f ~ %.2f ppm' %(f2M, f2m))
    st.write('Range for F1: %.2f ~ %.2f ppm' %(f1M, f1m))
    
    fig, ax = plt.subplots()
    ax.set_xlabel('F2, ppm')
    ax.set_ylabel('F1, ppm')
    arr = df.to_numpy()
    alpha = 0.5
    plt.contour(x, y, arr,
                 cmap = 'gray_r',
                 vmin = arr.mean(), vmax = (1-alpha) * arr.mean() + alpha * arr.max())
    st.write('\n Thumbnail spectrum is')
    fig


    st.write('Set the range for x-axis >_<')

    col1, col2 = st.columns(2)
    keyx1 = f"rangex_{i}"
    keyx2 = f"rangexx_{i}"
    keyy1 = f"rangey_{i}"
    keyy2 = f"rangeyy_{i}"
    with col1:
      x1 = st.number_input('Start point of F2(ppm)', f2m, f2M, f2m, key=keyx1)
      y1 = st.number_input('Start point of F1(ppm)', f1m, f1M, f1m, key=keyy1)
    with col2: 
      x2 = st.number_input('End point of F2(ppm)', f2m, f2M, f2M, key=keyx2)
      y2 = st.number_input('End point of F1(ppm)', f1m, f1M,f1M, key=keyy2)

  fig, ax = plt.subplots()
  ax.set_xlabel('F2, ppm')
  ax.set_ylabel('F1, ppm')

  arr = df.to_numpy()

  x_idx1 = np.argmin(np.absolute(x - x1))
  x_idx2 = np.argmin(np.absolute(x - x2))
  y_idx1 = np.argmin(np.absolute(y - y1))
  y_idx2 = np.argmin(np.absolute(y - y2))

  fig, ax = plt.subplots(figsize = (8,4))
  Xcrop = x[x_idx2:x_idx1]
  Ycrop = y[y_idx2:y_idx1]
  arrcrop = arr[y_idx2:y_idx1, x_idx2:x_idx1]
  ax.set_xlabel('F2, ppm')
  ax.set_ylabel('F1, ppm')

  alpha = 0.05
  plt.contour(Xcrop, Ycrop, arrcrop,
                 cmap = 'gray_r',
                 vmin = arr.mean(), vmax = (1-alpha) * arr.mean() + alpha * arr.max()) 

  ax.invert_xaxis()  
  ax.invert_yaxis()
  fig


def opt_fit():
  return


for i, uploaded_file in enumerate(uploaded_files):
    ## print order and information of files ##
    if i + 1 == 1:
      st.header(str(i + 1)+'st file')
    elif i + 1 == 2:
      st.header(str(i + 1)+'nd file')
    elif i + 1 == 3:
      st.header(str(i + 1),'rd file')
    else:
      st.header(str(i + 1),'th file')
        
    st.write(uploaded_file)
    
    ## set keys for iteration ##
    key = f"slider_{i}"
    key1 = f"slider2_{i}"
    key2 = f"check_{i}"
    key3 = f'opt_{i}'

    ## Choose processing option ##
    options = ['--------', '1D spectrum', '2D spectrum', 'Fitting']
    selected_option = st.selectbox("Select a processing option:", options, key = key3)
    
    if selected_option == '--------':
      st.write('--------')
    elif selected_option == '1D spectrum':
      opt_1d_spec()
    elif selected_option == '2D spectrum':
      if not os.path.exists('./tmp/data2d.txt'):
         print("Make a 2d data file.")
         read_param()
         
      opt_2d_spec()
    elif selected_option == 'Fitting':
      opt_fit()