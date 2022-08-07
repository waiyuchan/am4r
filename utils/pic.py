import matplotlib.pyplot as plt

# valid accuracy
x_axix = [1, 2, 3, 4, 5]
bert_base = [0.886, 0.924, 0.930, 0.936, 0.932]
bert_base_lstm = [0.871, 0.931, 0.936, 0.943, 0.940]
bert_base_bilstm = [0.922, 0.937, 0.924, 0.934, 0.947]
bert_large = [0.932, 0.940, 0.941, 0.942, 0.929]
bert_large_lstm = [0.875, 0.943, 0.941, 0.947, 0.944]
bert_large_bilstm = [0.937, 0.949, 0.951, 0.956, 0.955]
plt.title('Valid Accuracy Analysis')
plt.plot(x_axix, bert_base, color='green', label='BERT-Base')
plt.plot(x_axix, bert_base_lstm, color='red', label='BERT-Base-LSTM')
plt.plot(x_axix, bert_base_bilstm, color='yellow', label='BERT-Base-BiLSTM')
plt.plot(x_axix, bert_large, color='skyblue', label='BERT-Large')
plt.plot(x_axix, bert_large_lstm, color='blue', label='BERT-Large-LSTM')
plt.plot(x_axix, bert_large_bilstm, color='purple', label='BERT-Large-BiLSTM')
plt.legend()  # 显示图例
plt.xticks(x_axix)
plt.xlabel('epoches')
plt.ylabel('rate')
plt.show()

# valid loss
x_axix = [1, 2, 3, 4, 5]
bert_base = [0.049, 0.031, 0.026, 0.024, 0.026]
bert_base_lstm = [0.043, 0.021, 0.022, 0.022, 0.021]
bert_base_bilstm = [0.023, 0.022, 0.026, 0.028, 0.027]
bert_large = [0.039, 0.026, 0.023, 0.022, 0.024]
bert_large_lstm = [0.052, 0.027, 0.013, 0.006, 0.004]
bert_large_bilstm = [0.029, 0.019, 0.011, 0.005, 0.002]
plt.title('Valid Loss Analysis')
plt.plot(x_axix, bert_base, color='green', label='BERT-Base')
plt.plot(x_axix, bert_base_lstm, color='red', label='BERT-Base-LSTM')
plt.plot(x_axix, bert_base_bilstm, color='yellow', label='BERT-Base-BiLSTM')
plt.plot(x_axix, bert_large, color='skyblue', label='BERT-Large')
plt.plot(x_axix, bert_large_lstm, color='blue', label='BERT-Large-LSTM')
plt.plot(x_axix, bert_large_bilstm, color='purple', label='BERT-Large-BiLSTM')
plt.legend()  # 显示图例
plt.xticks(x_axix)
plt.xlabel('epoches')
plt.ylabel('rate')
plt.show()

# training accuracy
x_axix = [1, 2, 3, 4, 5]
bert_base = [0.683, 0.913, 0.944, 0.964, 0.975]
bert_base_lstm = [0.642, 0.914, 0.963, 0.982, 0.988]
bert_base_bilstm = [0.868, 0.956, 0.980, 0.990, 0.996]
bert_large = [0.769, 0.934, 0.954, 0.970, 0.982]
bert_large_lstm = [0.872, 0.943, 0.977, 0.989, 0.991]
bert_large_bilstm = [0.889, 0.959, 0.989, 0.992, 0.997]
plt.title('Training Accuracy Analysis')
plt.plot(x_axix, bert_base, color='green', label='BERT-Base')
plt.plot(x_axix, bert_base_lstm, color='red', label='BERT-Base-LSTM')
plt.plot(x_axix, bert_base_bilstm, color='yellow', label='BERT-Base-BiLSTM')
plt.plot(x_axix, bert_large, color='skyblue', label='BERT-Large')
plt.plot(x_axix, bert_large_lstm, color='blue', label='BERT-Large-LSTM')
plt.plot(x_axix, bert_large_bilstm, color='purple', label='BERT-Large-BiLSTM')
plt.legend()  # 显示图例
plt.xticks(x_axix)
plt.xlabel('epoches')
plt.ylabel('rate')
plt.show()

# training loss
x_axix = [1, 2, 3, 4, 5]
bert_base = [0.074, 0.037, 0.025, 0.018, 0.014]
bert_base_lstm = [0.075, 0.029, 0.014, 0.008, 0.005]
bert_base_bilstm = [0.038, 0.015, 0.008, 0.004, 0.003]
bert_large = [0.064, 0.033, 0.022, 0.016, 0.011]
bert_large_lstm = [0.052, 0.027, 0.013, 0.006, 0.004]
bert_large_bilstm = [0.029, 0.019, 0.011, 0.005, 0.002]
plt.title('Training Loss Analysis')
plt.plot(x_axix, bert_base, color='green', label='BERT-Base')
plt.plot(x_axix, bert_base_lstm, color='red', label='BERT-Base-LSTM')
plt.plot(x_axix, bert_base_bilstm, color='yellow', label='BERT-Base-BiLSTM')
plt.plot(x_axix, bert_large, color='skyblue', label='BERT-Large')
plt.plot(x_axix, bert_large_lstm, color='blue', label='BERT-Large-LSTM')
plt.plot(x_axix, bert_large_bilstm, color='purple', label='BERT-Large-BiLSTM')
plt.legend()  # 显示图例
plt.xticks(x_axix)
plt.xlabel('epoches')
plt.ylabel('rate')
plt.show()
