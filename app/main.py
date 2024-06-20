import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from get_data import get_list, get_data_realtime, get_data_his
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LR.extract_feature import predict_n_days


# Giao diện Streamlit
st.title('VNStock Realtime Data')
stock_dict = get_list().set_index('organ_name')['ticker'].to_dict()
selected_stock = st.selectbox('Chọn cổ phiếu', options=list(stock_dict.keys()))
# Nhập mã cổ phiếu
if selected_stock:
    st.write(f"Bạn đã chọn: {selected_stock} với id: {stock_dict[selected_stock]}")

# Nút để cập nhật dữ liệu
if st.button('Update Data'):
    if selected_stock:
        with st.spinner('Fetching data...'):
            close = get_data_realtime(stock_dict[selected_stock])
            st.write(close)
    else:
        st.write('Chưa chọn cổ phiếu')
df_hist = get_data_his(stock_dict[selected_stock])
# Tạo các ngày cho dữ liệu dự đoán, bỏ qua ngày thứ 7 và chủ nhật
predicted_close = predict_n_days(df_hist, n = 10)
last_date = df_hist['time'].iloc[-1]
pred_dates = pd.bdate_range(start=last_date, periods=predicted_close.shape[0] + 1, closed='right')

# Tạo DataFrame cho dữ liệu dự đoán
df_pred = pd.DataFrame({
    'time': pred_dates,
    'predicted_close': predicted_close
})
df_pred = pd.concat([
    pd.DataFrame({'time': [df_hist['time'].iloc[-1]], 'predicted_close': [df_hist['close'].iloc[-1]]}),
    df_pred
]).reset_index(drop=True)
print(df_pred)

st.title('Biểu đồ Candlestick và Dữ liệu Dự đoán')

# Tạo biểu đồ candlestick từ dữ liệu lịch sử
fig = go.Figure(data=[go.Candlestick(
    x=df_hist['time'],
    open=df_hist['open'],
    high=df_hist['high'],
    low=df_hist['low'],
    close=df_hist['close'],
    name='Candlestick'
)])
fig.add_trace(go.Scatter(
    x=df_hist['time'],
    y=df_hist['close'],
    mode='lines',
    name='Close (Lịch sử)'
))

# Thêm dữ liệu dự đoán vào biểu đồ
fig.add_trace(go.Scatter(
    x=df_pred['time'],
    y=df_pred['predicted_close'],
    mode='lines',
    name='Dự đoán Close',
    line=dict(color='red')
))

fig.update_layout(
    title='Biểu đồ Candlestick với Dữ liệu Dự đoán',
    xaxis_title='Ngày',
    yaxis_title='Giá',
    xaxis_rangeslider_visible=False
)

# Hiển thị biểu đồ trong Streamlit
st.plotly_chart(fig)
