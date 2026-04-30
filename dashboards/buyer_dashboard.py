import streamlit as st

def buyer_dashboard(state, district, crop, future_date, filtered, hist, pred_price, perc_change, recommendation, pred_source):
    st.header("💰 Buyer Dashboard")
    st.success(f"Predicted {crop} price on {future_date}: ₹{pred_price:,.2f}")
    st.info(f"Recommendation: {recommendation}")
    st.caption(pred_source)
    st.subheader("Historical Modal Price Trend")
    st.line_chart(hist.set_index("Date")["Modal Price"])
    st.subheader("Predicted Price for Selected Date")
    st.write(f"Predicted Price: ₹{pred_price:,.2f} | Change: {perc_change:.2f}%")
