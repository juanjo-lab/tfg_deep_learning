########                                            LIBRERIAS
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
########################################################################################################################
#python manage.py runserver
########                                            DATOS
########################################################################################################################
MSE_Hibrido_LSTM = 1179
MSE_Ficticio_LSTM = 2217.030945328409
MSE_Real_LSTM = 1338.9951339176728
LSTM=[MSE_Ficticio_LSTM, MSE_Hibrido_LSTM, MSE_Real_LSTM]

MSE_Real_LSTM_5 = 1021.7333638181477
MSE_Ficticio_LSTM_5 = 2217.030945328409
MSE_Hibrido_LSTM_5 = 1015.9757744178372
LSTM_optm=[MSE_Ficticio_LSTM_5, MSE_Hibrido_LSTM_5, MSE_Real_LSTM_5]


MSE_Hibrido_SARIMAX = 1033.900
MSE_Ficticio_SARIMAX = 1033.900
MSE_Real_SARIMAX = 1033.900
SARIMAX=[MSE_Ficticio_SARIMAX, MSE_Hibrido_SARIMAX, MSE_Real_SARIMAX]
########################################################################################################################

########                                            REPRESENTACION
########################################################################################################################
width=0.2
fig, ax = plt.subplots()
xas=['FICTICIO','HIBRIDO','REAL']
x=np.arange((len(xas)))

#rects1 = ax.bar(x-width/2,LSTM,width,label='LSTM_90_epoch')
rects2 = ax.bar(x+width/2,SARIMAX,width,label='SARIMAX')
rects3 = ax.bar(x-width,LSTM_optm,width,label='LSTM_5_epoch_opt')

ax.set_ylabel('MSE')
ax.set_xlabel("FICTICIO                      HIBRIDO                       REAL")
ax.set_title('Comparación entre LSTM 5 epoch vs SARIMAX')
ax.grid(color='grey', linestyle='dashed', linewidth=0.5)

ax.legend()
plt.show()
########################################################################################################################